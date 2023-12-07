#!/usr/bin/env python

import os
import glob
import sys
from datetime import datetime
import cftime
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import dask
import dask.array as da
from dask.utils import M

# fmt:off
HEADLEN     = 1024  # ヘッダのデータ長
DELIMLEN    = 4     # デリミタのデータ長
HDITEMS     = ['IDFM',  'DSET',  'ITEM',  'EDIT1', 'EDIT2', 'EDIT3',
               'EDIT4', 'EDIT5', 'EDIT6', 'EDIT7', 'EDIT8', 'FNUM',
               'DNUM',  'TITL1', 'TITL2', 'UNIT',  'ETTL1', 'ETTL2',
               'ETTL3', 'ETTL4', 'ETTL5', 'ETTL6', 'ETTL7', 'ETTL8',
               'TIME',  'UTIM',  'DATE',  'TDUR',  'AITM1', 'ASTR1',
               'AEND1', 'AITM2', 'ASTR2', 'AEND2', 'AITM3', 'ASTR3',
               'AEND3', 'DFMT',  'MISS',  'DMIN',  'DMAX',  'DIVS',
               'DIVL',  'STYP',  'OPT1',  'OPT2',  'OPT3',  'MEMO1',
               'MEMO2', 'MEMO3', 'MEMO4', 'MEMO5', 'MEMO6', 'MEMO7',
               'MEMO8', 'MEMO9', 'MEMO10','MEMO11','MEMO12','CDATE',
               'CSIGN', 'MDATE', 'MSIGN', 'SIZE']                       # ヘッダ項目
INTITEMS    = ['FNUM', 'DNUM', 'TIME', 'TDUR', 'ASTR1', 'AEND1', 
               'ASTR2', 'AEND2', 'ASTR3', 'AEND3', 'STYP', 'SIZE']      # 整数のヘッダ項目
FLTITEMS    = ['MISS', 'DMIN', 'DMAX', 'DIVS', 'DIVL']                  # 実数のヘッダ項目

NDTYP       = {'UR4':'f4', 'UR8':'f8', 'URY16':'u2', 'MR4':'f4'}        # フォーマットごとのデータ型

GTAXDIR     = os.environ.get('GTAXDIR') # 軸データの場所

DIMNAME     = {'latitude':'lat', 'longitude':'lon', 'pressure':'plev'}  # 軸名の短縮形
# fmt:on


class MultiFileGtool3:
    def __init__(self, path, omit="neither"):
        self.paths = self.list_paths(path, omit)
        self.datainfo = Gtool3(self.paths[0]).datainfo
        return

    def list_paths(self, path, omit):
        if os.path.isfile(path):
            path = [path]
        else:
            path = glob.glob(path)
            if len(path) == 0:
                raise ValueError("No Files Found!")
            path.sort()
            if omit == "neither":
                pass
            elif omit == "max":
                # 最後のファイルを使わない
                # 書き込み中の場合など
                path = path[:-1]
            elif omit == "min":
                # 最初のファイルを使わない
                # 初期条件が入っている場合など
                path = path[1:]
            elif omit == "both":
                path = self.path[1:-1]
            else:
                raise ValueError(
                    'Omit must be either "max", "min", "both", or "neither".'
                )
        return path

    def mfopen(self, unstack=True):
        files = [Gtool3(path) for path in self.paths]
        data = [file.open_data() for file in files]
        data = da.concatenate(data)
        time = [file.make_time_ax() for file in files]
        time = np.concatenate(time)
        data = xr.DataArray(
            name=self.datainfo["item"],
            data=data,
            dims=["time", "stax"],
            coords={"time": time, "stax": self.datainfo["stax"]},
            attrs={"title": self.datainfo["title"], "unit": self.datainfo["unit"]},
        )
        data = data.where(data != self.datainfo["miss"])
        if unstack or (self.datainfo["dfmt"] == "URY16"):
            data = data.unstack()
        if self.datainfo["dfmt"] == "URY16":
            coefs = [file.open_coef() for file in files]
            coefs = da.concatenate(coefs)
            zdim = self.datainfo["dims"][0]
            offset = xr.DataArray(
                data=coefs[:, :, 0],
                dims=["time", zdim],
                coords={"time": time, zdim: self.datainfo["coords"][zdim]},
            )
            scale = xr.DataArray(
                data=coefs[:, :, 1],
                dims=["time", zdim],
                coords={"time": time, zdim: self.datainfo["coords"][zdim]},
            )
            data = scale * data + offset
        if unstack:
            data = data.squeeze()
        return data


class Gtool3:
    def __init__(self, path, datainfo=None):
        self.path = path
        if datainfo is None:
            self.parse_file()
        else:
            self.datainfo = datainfo
        self.file = self.mmap_dask_array()
        return

    def parse_file(self):
        self.head = self.read_header()
        self.datainfo = self.set_datainfo1()
        if self.datainfo["dfmt"] == "MR4":
            self.read_mask()
            self.datainfo = pd.concat(
                [self.datainfo, pd.Series({"mask": self.mask}), self.set_datainfo2()]
            )
        else:
            self.datainfo = pd.concat([self.datainfo, self.set_datainfo2()])
        self.open_axs()
        self.make_stacked_axs()
        return

    def interpret_delim(self):
        # デリミタを使ってエンディアンを区別する
        b_delim = np.memmap(self.path, dtype=">u4", mode="r", shape=1)
        l_delim = np.memmap(self.path, dtype="<u4", mode="r", shape=1)
        if b_delim == HEADLEN:
            # デリミタをbig endianで読んでみた結果がHEADLENと一致した
            endian = ">"
        elif l_delim == HEADLEN:
            # デリミタをlittle endianで読んでみた結果がHEADLENと一致した
            endian = "<"
        else:
            raise ValueError("DELIM does not match HEADLEN!")
        return endian

    def read_header(self):
        # 最初のブロックのヘッダを読む
        head = np.memmap(
            self.path, dtype="S16", mode="r", offset=DELIMLEN, shape=HEADLEN // 16
        )
        head = [h.decode().strip() for h in head]
        head = dict(zip(HDITEMS, head))
        for item in INTITEMS:
            # 整数の項目を文字列から整数に直す
            # ただし空欄はそのまま
            head[item] = int(head[item]) if head[item] != "" else ""
        for item in FLTITEMS:
            # 実数の項目を文字列から実数に直す
            # ただし空欄はそのまま
            head[item] = float(head[item]) if head[item] != "" else ""
        head = pd.Series(head)
        return head

    def set_datainfo1(self):
        # ヘッダの情報等をもとに変数を設定する
        filelen = os.path.getsize(self.path)
        endian = self.interpret_delim()
        dfmt = self.head.DFMT
        shape = [
            self.head["AEND" + str(i + 1)] - self.head["ASTR" + str(i + 1)] + 1
            for i in range(3)
        ][::-1]
        size = shape[0] * shape[1] * shape[2]
        axnm = [self.head["AITM" + str(i + 1)] for i in range(3)][::-1]
        axsel = [
            slice(self.head["ASTR" + str(i + 1)] - 1, self.head["AEND" + str(i + 1)])
            for i in range(3)
        ][::-1]
        ndtyp = endian + NDTYP[dfmt]
        rlen = int(ndtyp[-1])  # ビット数
        miss = (
            self.head["MISS"]
            if dfmt != "URY16"
            else int.from_bytes(
                b"\xff\xff", byteorder=("big" if endian == ">" else "little")
            )
        )
        item = self.head["ITEM"]
        title = self.head["TITL1"] + self.head["TITL2"]
        unit = self.head["UNIT"]
        coefsize = 2 * shape[0] if dfmt == "URY16" else None
        coeflen = 8 * coefsize if dfmt == "URY16" else None
        datainfo1 = pd.Series(
            {
                "filelen": filelen,
                "endian": endian,
                "dfmt": dfmt,
                "shape": shape,
                "size": size,
                "axnm": axnm,
                "axsel": axsel,
                "ndtyp": ndtyp,
                "rlen": rlen,
                "miss": miss,
                "item": item,
                "title": title,
                "unit": unit,
            }
        )
        if dfmt == "URY16":
            datainfo1 = pd.concat(
                [datainfo1, pd.Series({"coefsize": coefsize, "coeflen": coeflen})]
            )
        return datainfo1

    def read_mask(self):
        dtype = "u1"
        offset = 3 * DELIMLEN + HEADLEN + 12
        shape = self.datainfo["size"] // 8
        mask = np.memmap(self.path, dtype=dtype, offset=offset, shape=shape)
        mask = np.unpackbits(mask)
        mask = mask.astype(int)
        mask = ma.masked_where(mask == 0, mask)
        mask = mask.astype(self.datainfo.ndtyp)
        self.mask = mask
        return

    def set_datainfo2(self):
        # マスクの情報まで含めて変数を設定する
        if self.datainfo["dfmt"] == "MR4":
            datasize = np.count_nonzero(self.mask)
        else:
            datasize = self.datainfo["size"]
        datalen = datasize * self.datainfo["rlen"]
        # ブロック１つあたりの長さ
        if self.datainfo["dfmt"] == "URY16":
            blen = 6 * DELIMLEN + HEADLEN + self.datainfo["coeflen"] + datalen
        elif self.datainfo["dfmt"] == "MR4":
            blen = (
                3 * DELIMLEN + HEADLEN + 2 * 12 + self.datainfo["size"] // 8 + datalen
            )
        else:
            blen = 4 * DELIMLEN + HEADLEN + datalen
        if self.datainfo["filelen"] % blen != 0:
            # raise ValueError('BLEN is not correct!')
            print("filelen/blen = ", self.datainfo["filelen"] / blen)
        nblocks = self.datainfo["filelen"] // blen
        datainfo2 = pd.Series(
            {"datasize": datasize, "datalen": datalen, "blen": blen, "nblocks": nblocks}
        )
        return datainfo2

    def read_chunks(self):
        chunks = [self.mmap_dask_array(path) for path in self.paths]
        chunks = da.concatenate(chunks)
        self.chunks = chunks
        return

    def open_axs(self):
        coords = {}
        for i in range(3):
            path = GTAXDIR + "/GTAXLOC." + self.datainfo["axnm"][i]
            ax = Gtool3Ax(path).open()[self.datainfo["axsel"][i]]
            coords[ax.name] = ax
        self.datainfo = pd.concat(
            [self.datainfo, pd.Series({"coords": coords, "dims": list(coords.keys())})]
        )
        return

    def make_stacked_axs(self):
        if self.datainfo["dfmt"] == "MR4":
            data = self.datainfo["mask"].reshape(self.datainfo["shape"])
        else:
            data = da.ones(self.datainfo["shape"])
        data = xr.DataArray(
            data=data, dims=self.datainfo["dims"], coords=self.datainfo["coords"]
        )
        data = data.stack(stax=self.datainfo["dims"])
        if self.datainfo["dfmt"] == "MR4":
            data = data.dropna(dim="stax")
        stax = data.stax
        self.datainfo = pd.concat([self.datainfo, pd.Series({"stax": stax})])
        return

    def make_time_ax(self):
        st = DELIMLEN + 16 * 26
        time = self.file[:, st : st + 16].view("S16")
        time = da.concatenate(time)
        time = time.compute()
        _time = np.empty(time.shape, dtype="O")
        for i, t in enumerate(time):
            dt = datetime.strptime(t.decode().strip(), "%Y%m%d %H%M%S")
            _time[i] = cftime.datetime(
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                dt.minute,
                dt.second,
                calendar="standard",
            )
        return _time

    def open(self, unstack=True):
        data = self.open_data()
        time = self.make_time_ax()
        data = xr.DataArray(
            name=self.datainfo["item"],
            data=data,
            dims=["time", "stax"],
            coords={"time": time, "stax": self.datainfo["stax"]},
        )
        data = data.where(data != self.datainfo["miss"])
        if unstack or (self.datainfo["dfmt"] == "URY16"):
            data = data.unstack()
        if self.datainfo["dfmt"] == "URY16":
            coef = self.open_coef()
            zdim = self.datainfo["dims"][0]
            offset = xr.DataArray(
                data=coef[:, :, 0],
                dims=["time", zdim],
                coords={"time": time, zdim: self.datainfo["coords"][zdim]},
            )
            scale = xr.DataArray(
                data=coef[:, :, 1],
                dims=["time", zdim],
                coords={"time": time, zdim: self.datainfo["coords"][zdim]},
            )
            self.scale = scale
            self.offset = offset
            self.test0 = data
            data = scale * data + offset
            self.test1 = data
        if unstack:
            data = data.squeeze()
        data = data.assign_attrs(
            {"title": self.datainfo["title"], "unit": self.datainfo["unit"]}
        )
        data = data.rename(self.datainfo['item'])
        return data

    def open_data(self):
        st = self.datainfo["blen"] - (self.datainfo["datalen"] + DELIMLEN)
        data = self.file[:, st : st + self.datainfo["datalen"]]
        data = data.view(self.datainfo.ndtyp)
        return data

    def open_coef(self):
        # st = 5*DELIMLEN + HEADLEN
        st = 3 * DELIMLEN + HEADLEN
        coef = self.file[:, st : st + self.datainfo["coeflen"]]
        coef = coef.view(self.datainfo["endian"] + "f8")
        coef = coef.reshape([self.datainfo["nblocks"], self.datainfo["shape"][0], 2])
        return coef

    def mmap_load_chunks(self):
        return np.memmap(
            self.path, dtype="S1", mode="r", shape=self.datainfo["filelen"]
        )

    def mmap_dask_array(self):
        load = dask.delayed(self.mmap_load_chunks)
        chunk = da.from_delayed(
            load(), shape=[self.datainfo["filelen"]], dtype="S1"
        ).reshape([self.datainfo["nblocks"], self.datainfo["blen"]])
        return chunk


class Gtool3Ax(Gtool3):
    def parse_file(self):
        self.head = self.read_header()
        self.datainfo = self.set_datainfo1()
        if self.datainfo["dfmt"] == "MR4":
            self.read_mask()
            self.datainfo = pd.concat(
                [self.datainfo, pd.Series({"mask": self.mask}), self.set_datainfo2()]
            )
        else:
            self.datainfo = pd.concat([self.datainfo, self.set_datainfo2()])
        self.file = self.mmap_dask_array()
        return

    def open(self):
        if self.datainfo["title"] in DIMNAME:
            title = DIMNAME[self.datainfo["title"]]
        else:
            title = self.datainfo["title"]
        data = self.open_data()[0, :].compute()
        if (sys.byteorder == "little" and data.dtype.byteorder == ">") or (
            sys.byteorder == "big" and data.dtype.byteorder == "<"
        ):
            data = data.byteswap().newbyteorder()
        data = xr.DataArray(
            name=title,
            data=data,
            dims=[title],
            coords={title: data},
            attrs={
                "item": self.datainfo["item"],
                "title": self.datainfo["title"],
                "unit": self.datainfo["unit"],
            },
        )
        return data
