#!/usr/bin/env python

from datetime import datetime
import numpy as np
import xarray as xr
import os
import glob
from concurrent.futures import ProcessPoolExecutor

HEADLEN     = 1024   # ヘッダのデータ長
DELIMLEN    = 4   # 区切りのデータ長
HDITEMS     = ['IDFM', 'DSET', 'ITEM', 'EDIT1', 'EDIT2', 'EDIT3', 'EDIT4', 'EDIT5',
               'EDIT6', 'EDIT7', 'EDIT8', 'FNUM', 'DNUM', 'TITL1', 'TITL2', 'UNIT',
               'ETTL1', 'ETTL2', 'ETTL3', 'ETTL4', 'ETTL5', 'ETTL6', 'ETTL7', 'ETTL8',
               'TIME', 'UTIM', 'DATE', 'TDUR', 'AITM1', 'ASTR1', 'AEND1', 'AITM2',
               'ASTR2', 'AEND2', 'AITM3', 'ASTR3', 'AEND3', 'DFMT', 'MISS', 'DMIN',
               'DMAX', 'DIVS', 'DIVL', 'STYP', 'OPT1', 'OPT2', 'OPT3', 'MEMO1',
               'MEMO2', 'MEMO3', 'MEMO4', 'MEMO5', 'MEMO6', 'MEMO7', 'MEMO8', 'MEMO9',
               'MEMO10', 'MEMO11', 'MEMO12', 'CDATE', 'CSIGN', 'MDATE', 'MSIGN', 'SIZE']   # ヘッダの項目
INTITEMS    = ['FNUM', 'DNUM', 'TIME', 'TDUR', 'ASTR1', 'AEND1', 
               'ASTR2', 'AEND2', 'ASTR3', 'AEND3', 'STYP', 'SIZE']   # 整数のヘッダ項目
FLTITEMS    = ['MISS', 'DMIN', 'DMAX', 'DIVS', 'DIVL']   # 実数のヘッダ項目

RLEN        = {'UR4':4, 'UR8':8, 'URY16':2, 'MR4':4}   # フォーマットごとのデータ長
NDTYP       = {'UR4':'>f4', 'UR8':'>f8', 'URY16':'>u2', 'MR4':'>f4'}   # フォーマットごとのデータ型

GTAXDIR     = os.environ.get('GTAXDIR')   # 軸データの場所

DIMNAME     = {'latitude':'lat', 'longitude':'lon', 'pressure':'plev'}   # 軸名の短縮

class Gtool3():

    def __init__(self, path, axs=None, unstack=True, squeeze=True):
        self.path = path   # ファイルの場所
        self.axs = axs   # 軸情報
        self.unstack = unstack   # Trueならunstackしたデータを返す
        self.squeeze = squeeze   # Trueなら長さ１の軸を消去したデータを返す
        self.file = np.memmap(self.path, 'S1', 'r')   # ファイル
        self.parse_file()   # ファイルの基礎情報を取得する
        return

    def parse_file(self):
        self.pos = 0   # ファイルの頭から順に読んでいく
        self.move(DELIMLEN)
        head = self.read_header()
        self.set_values(head)

        if self.axs is None:
            if self.dfmt == 'MR4':
                self.move(2*DELIMLEN+12)
                mask = self.read_mask()
            else:
                mask = np.ones(self.size)
            mask = mask.reshape(self.sel[:,2])

            axs, wgt = self.get_axs()
            self.dims = [ax.name for ax in axs]

            mask = xr.DataArray(
                data = mask, 
                dims = self.dims, 
                coords = axs
            )
            mask = mask.stack(stdims = self.dims)
            mask = mask.where(mask==1, drop=True)
            self.axs = mask.stdims
            if wgt:
                self.wgt = xr.merge(wgt)
            else:
                self.wgt = None

        if self.dfmt == 'MR4':
            self.size_masked = len(self.axs)
        return

    def set_values(self, head):
        self.item = head['ITEM']
        self.dfmt = head['DFMT']
        self.rlen = RLEN[self.dfmt]
        self.ndtyp = NDTYP[self.dfmt]
        self.size = head['SIZE']
        self.miss = head['MISS']
        self.axname = [head['AITM' + str(i)] for i in range(3, 0, -1)]
        self.sel = np.array([[
            head['ASTR' + str(i)], 
            head['AEND' + str(i)], 
            head['AEND' + str(i)] - head['ASTR' + str(i)] + 1
        ] for i in range(3, 0, -1)])   # z, y, x軸の始点、終点、長さ
        self.title = head['TITL1'] + head['TITL2']
        self.unit = head['UNIT']

    def open(self):
        self.time_list = []
        self.data_list = []
        self.scale_list = []
        self.offset_list = []

        self.end = len(self.file)
        self.pos = 0

        self.move(DELIMLEN)
        while self.pos <= self.end:
            head = self.read_header()
            time = datetime.strptime(head['DATE'], '%Y%m%d %H%M%S')
            self.time_list.append(time)
            self.move(2*DELIMLEN)
            if self.dfmt == 'URY16':   # URY16の場合、先にスケールとゲタを読み込む。
                coef = self.read_coef()
                self.scale_list.append(coef[:,1])
                self.offset_list.append(coef[:,0])
                self.move(2*DELIMLEN)
                self.miss = int.from_bytes(b'\xff\xff', byteorder='big')
            if self.dfmt == 'MR4':   # MR4の場合、マスクを読み込む。
                self.move(12)
                self.move(self.size//8)
                self.move(12)
                self.data = self.read_masked()
                self.move(DELIMLEN)
            else:   # その他の場合、そのまま読み込む。
                self.data = self.read_data()
                self.move(2*DELIMLEN)
            self.data = self.remove_miss()
            self.data_list.append(self.data)

        self.data_list = np.array(self.data_list)
        self.data_list = self.data_list.byteswap().newbyteorder()   # force native byteorder
        self.time_list = np.array(self.time_list, dtype=np.datetime64)
        self.time_list = self.time_list.byteswap().newbyteorder()   # force native byteorder
        self.scale_list = np.array(self.scale_list)
        self.scale_list = self.scale_list.byteswap().newbyteorder()   # force native byteorder
        self.offset_list = np.array(self.offset_list)
        self.offset_list = self.offset_list.byteswap().newbyteorder()   # force native byteorder
        self.da = self.to_DataArray()
        if self.dfmt == 'URY16':
            self.da = self.denorm_data()
        if self.squeeze:
            self.da = self.da.squeeze()
        return self.da

    def move(self, len):
        self.pos += len
        return

    def read_and_move(self, len):
        sel = self.file[self.pos:self.pos+len]
        self.pos += len
        return sel

    def read_header(self):
        values = self.read_and_move(HEADLEN).view('S16')
        values = [v.decode().strip() for v in values]
        head = dict(zip(HDITEMS, values))
        for item in INTITEMS:
            head[item] = int(head[item]) if head[item] != '' else None
        for item in FLTITEMS:
            head[item] = float(head[item]) if head[item] != '' else None
        return head

    def read_data(self):
        data = self.read_and_move(self.size*self.rlen)
        data = np.frombuffer(data, dtype=self.ndtyp)
        data = data.byteswap().newbyteorder()   # force native byteorder
        return data

    def read_coef(self):
        coef = self.read_and_move(8*2*self.sel[0,1])
        coef = np.frombuffer(coef, dtype='>f8')
        coef = coef.byteswap().newbyteorder()   # force native byteorder
        coef = coef.reshape(self.sel[0,1], 2)
        return coef

    def read_mask(self):
        mask = self.read_and_move(self.size//8)
        mask = np.frombuffer(mask, dtype=np.uint8)
        mask = mask.byteswap().newbyteorder()   # force native byteorder
        mask = np.unpackbits(mask)
        mask = mask.astype(int)
        return mask

    def read_masked(self):
        data = self.read_and_move(self.size_masked*self.rlen)
        data = np.frombuffer(data, dtype=self.ndtyp)
        data = data.byteswap().newbyteorder()   # force native byteorder
        return data

    def denorm_data(self):
        zdim = self.da.dims[1]
        zcoord = self.da[zdim]
        scale = xr.DataArray(
            data = self.scale_list, 
            dims = ['time', zdim], 
            coords = {'time':self.time_list, zdim:zcoord}
        ).chunk(chunks='auto')
        offset = xr.DataArray(
            data = self.offset_list, 
            dims = ['time', zdim], 
            coords = {'time':self.time_list, zdim:zcoord}
        ).chunk(chunks='auto')
        attrs = self.da.attrs
        da = self.da*scale + offset
        da = da.assign_attrs(attrs)
        return da

    def remove_miss(self):
        data = np.where(self.data==self.miss, np.nan, self.data)
        return data
    
    def get_axs(self):
        axs = [Gtool3Ax(GTAXDIR + '/GTAXLOC.' + self.axname[i], slice(self.sel[i,0] - 1, self.sel[i,1])).open() for i in range(3)]
        wgts = []
        for i in range(3):
            path = GTAXDIR + '/GTAXWGT.' + self.axname[i]
            if os.path.exists(path):
                wgts.append(Gtool3Wgt(path, axs[i], slice(self.sel[i,0] - 1, self.sel[i,1])).open())
        return axs, wgts
    
    def to_DataArray(self):
        da = xr.DataArray(
            data = self.data_list, 
            dims = ['time', 'stdims'], 
            coords = {'time':self.time_list, 'stdims':self.axs}, 
            name = self.item, 
            attrs = {'title':self.title, 'unit':self.unit}
        )
        if self.unstack:
            da = da.unstack()
        da = da.chunk(chunks='auto')
        return da

class Gtool3Ax(Gtool3):

    def __init__(self, path, sel=False):
        super().__init__(path)
        self.sel = sel
        return
    
    def parse_file(self):
        self.pos = 0
        self.move(DELIMLEN)
        head = self.read_header()
        self.set_values(head)
        return

    def open(self):
        self.pos = 0

        self.move(DELIMLEN)
        self.head = self.read_header()
        self.move(2*DELIMLEN)
        self.data = self.read_data()
        if self.sel:
            self.data = self.data[self.sel]
        self.DataArray = self.to_DataArray()
        return self.DataArray

    def to_DataArray(self):
        title = self.title
        if title in DIMNAME.keys():
            title = DIMNAME[title]
        da = xr.DataArray(
            data = self.data,
            dims = [title],
            name = title, 
            attrs = {'unit':self.unit}
        )
        da = da.chunk(chunks='auto')
        return da

class Gtool3Wgt(Gtool3Ax):

    def __init__(self, path, ax, sel=False):
        super().__init__(path, sel=sel)
        self.ax = ax
        return
    
    def to_DataArray(self):
        title = self.title
        if title in DIMNAME.keys():
            title = DIMNAME[title]
        da = xr.DataArray(
            data = self.data, 
            dims = [title], 
            coords = {title:self.ax.to_numpy()}, 
            name = 'wgt_' + title, 
            attrs = {'unit':self.unit}
        )
        da = da.chunk(chunks='auto')
        return da

class MultiFileGtool3():

    def __init__(self, path):
        self.paths = glob.glob(path)
        self.paths.sort()
        self.da0 = self.open_axs_and_mask(self.paths[1])
        return
    
    def open_axs_and_mask(self, path):
        gt3 = Gtool3(path, squeeze=False, unstack=True)
        da = gt3.open()
        self.axs = gt3.axs
        if gt3.wgt:
            self.wgt = gt3.wgt
        return da

    def sfopen(self, path):
        print('\rreading ' + path, end='')
        try:
            gt3 = Gtool3(path, axs=self.axs, squeeze=False, unstack=True)
            da = gt3.open()
            return da
        except:
            return

    def mfopen(self, max_workers=None):
        self.n = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            das = list(executor.map(self.sfopen, self.paths))
        # das = [self.sfopen(path) for path in self.paths[1:]]
        print('')
        das = [da for da in das if da is not None]
        da = xr.concat(das, dim='time', combine_attrs='identical')
        da = da.squeeze().unstack()
        da = da.chunk(chunks='auto')
        return da
