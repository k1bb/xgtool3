#!/usr/bin/env python

import os, glob
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import dask
import dask.array as da

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

RLEN        = {'UR4':4, 'UR8':8, 'URY16':2, 'MR4':4}                    # フォーマットごとのデータ長
NDTYP       = {'UR4':'f4', 'UR8':'f8', 'URY16':'u2', 'MR4':'f4'}        # フォーマットごとのデータ型

GTAXDIR     = os.environ.get('GTAXDIR') # 軸データの場所

DIMNAME     = {'latitude':'lat', 'longitude':'lon', 'pressure':'plev'}  # 軸名の短縮形

class Gtool3():

    def __init__(self, paths, omit='neither'):
        self.paths = glob.glob(paths)
        self.paths.sort()
        if omit == 'max':
            # 最後のファイルを使わない
            # 書き込み中の場合など
            self.paths = self.paths[:-1]
        elif omit == 'min':
            # 最初のファイルを使わない
            # 初期条件が入っている場合など
            self.paths = self.paths[1:]
        elif omit == 'both':
            self.paths = self.paths[1:-1]
        self.files = [np.memmap(path, 'S1', 'r') for path in self.paths]
        self.parse_file()
        return
        
    def parse_file(self):
        self.interpret_delim()
        self.set_values1(self.read_header())
        self.open_axs()
        self.make_stacked_axs()
        self.set_values2()
        self.make_time_ax()
        return

    def interpret_delim(self):
        # デリミタを使ってエンディアンを区別する
        delim = self.file(0, 0, DELIMLEN)
        delim_b = np.frombuffer(delim, dtype='>u4')[0]   # big endianを仮定して読む
        delim_l = np.frombuffer(delim, dtype='<u4')[0]   # little endianを仮定して読む
        if delim_b == HEADLEN:
            self.endian = '>'   # big endianで読んだ結果がHEADLENと一致したら接頭辞として'>'
        elif delim_l == HEADLEN:
            self.endian = '<'   # little endianで読んだ結果が...
        else:
            raise ValueError('DELIM does not match HEADLEN!')
        return
    
    def read_header(self, i=0, j=0):
        # i番目のファイルのj番目のブロックのヘッダを読む
        if j==0:
            st = DELIMLEN
        else:
            try:
                st = j*self.blen + DELIMLEN
            except:
                # blenがない場合はエラー　
                raise ValueError('Block Length is unknown.')
        values = self.file(i, st, HEADLEN).view('S16')
        values = [v.decode().strip() for v in values]
        values = dict(zip(HDITEMS, values))
        for item in INTITEMS:
            # 整数の項目を文字列から整数に直す
            # ただし空欄はそのまま
            values[item] = int(values[item]) if values[item] != '' else ''
        for item in FLTITEMS:
            # 実数の項目を文字列から実数に直す
            # ただし空欄はそのまま
            values[item] = float(values[item]) if values[item] != '' else ''
        values = pd.Series(values)
        return values
    
    def set_values1(self, values):
        # ヘッダの情報をもとに変数を設定する
        self.nfiles = len(self.paths)
        self.filelen = len(self.file())
        self.dfmt = values.DFMT
        self.shape = [
            values['AEND'+str(i+1)] - values['ASTR'+str(i+1)] + 1 for i in range(3)
        ][::-1]
        if values['SIZE'] == '':
            self.size = self.shape[0]*self.shape[1]*self.shape[2]
        else:
            self.size = values['SIZE']
        self.axnm = [values['AITM'+str(i+1)] for i in range(3)][::-1]
        self.axsel = [slice(values['ASTR'+str(i+1)]-1, values['AEND'+str(i+1)]) for i in range(3)][::-1]
        self.rlen = RLEN[self.dfmt]
        self.ndtyp = self.endian + NDTYP[self.dfmt]
        self.miss = (values['MISS'] if self.dfmt != 'URY16' else
                     int.from_bytes(b'\xff\xff', byteorder=('big' if self.endian=='>' else 'little')))
        self.item = values['ITEM']
        self.title = values['TITL1'] + values['TITL2']
        self.unit = values['UNIT']
        self.coeflen = 8*2*self.shape[0]
        return
    
    def set_values2(self):
        # マスクの情報まで含めて変数を設定する
        self.datasize = len(self.stax)
        self.datalen = self.datasize*self.rlen
        # ブロック１つあたりの長さ
        if self.dfmt == 'URY16':
            self.blen = 6*DELIMLEN + HEADLEN + self.coeflen + self.datalen
        elif self.dfmt == 'MR4':
            self.blen = 3*DELIMLEN + HEADLEN + 2*12 + self.size//8 + self.datalen
        else:
            self.blen = 4*DELIMLEN + HEADLEN + self.datalen
        if self.filelen%self.blen != 0:
            # raise ValueError('BLEN is not correct!')
            print('filelen/blen = ', self.filelen/self.blen)
        self.nblocks = self.filelen//self.blen
        return
    
    def make_stacked_axs(self):
        # x, y, z軸をスタックした軸データを作る
        if self.dfmt == 'MR4':
            # MR4のデータではマスクを読み取る
            mask = self.file(
                0, 
                3*DELIMLEN + HEADLEN + 12, 
                self.size//8
            )
            mask = np.frombuffer(mask, dtype=np.uint8)
            mask = np.unpackbits(mask)
            mask = mask.astype(int)
            mask = mask.reshape(self.shape)
        else:
            # それ以外の場合、np.onesを仮にマスクとして使用する
            mask = np.ones(self.shape)
        mask = xr.DataArray(
            name = 'mask', 
            data = mask, 
            dims = self.dims, 
            coords = {self.dims[i]:self.axs[i] for i in range(3)}
        )
        mask = mask.stack(stax=self.dims)
        if self.dfmt == 'MR4':
            mask = mask.where(mask==1, drop=True)
        self.mask = mask
        self.stax = mask.stax
        return
    
    def make_time_ax(self):
        tmax = []
        for i in range(self.nfiles):
            for j in range(self.nblocks):
                values = self.read_header(i, j)
                time = values['DATE']
                time = datetime.strptime(time, '%Y%m%d %H%M%S')
                time = np.datetime64(time)
                tmax.append(time)
        tmax = np.array(tmax)
        self.tmax = tmax
        return
    
    def open(self, unstack=True):
        chunks = []
        for i in range(self.nfiles):
            for j in range(self.nblocks):
                chunks.append(self.read_data_chunk(i, j))
        chunks = da.stack(chunks)
        data = xr.DataArray(
            name = self.item, 
            data = chunks, 
            dims = ['time', 'stax'], 
            coords = {'time':self.tmax, 'stax':self.stax}, 
            attrs = {'title':self.title, 'unit':self.unit, 'item':self.item}
        )
        data = data.where(data != self.miss)
        if unstack:
            data = data.unstack()
        if self.dfmt == 'URY16':
            self.open_coef()
            if not unstack:
                print('You need to manually apply coefs after unstacking the DataArray.')
            else:
                data = self.offset + data*self.scale
        for i in range(3):
            for attr in self.axs[i].attrs:
                data[self.dims[i]].attrs[attr] = self.axs[i].attrs[attr]
        data = data.squeeze()
        data = data.chunk({'time':'auto'})
        return data
    
    def open_coef(self):
        chunks = []
        for i in range(self.nfiles):
            for j in range(self.nblocks):
                chunks.append(self.read_coef_chunk(i, j))
        chunks = da.stack(chunks)
        chunks = chunks.reshape([self.nblocks*self.nfiles, self.shape[0], 2])
        zdim, zax = self.dims[0], self.axs[0]
        self.offset = xr.DataArray(
            name = 'offset', 
            data = chunks[:,:,0], 
            dims = ['time', zdim], 
            coords = {'time':self.tmax, zdim:zax}
        )
        self.scale = xr.DataArray(
            name = 'scale', 
            data = chunks[:,:,1], 
            dims = ['time', zdim], 
            coords = {'time':self.tmax, zdim:zax}
        )
        return
    
    def open_axs(self):
        axs = []
        for i in range(3):
            path = GTAXDIR + '/GTAXLOC.' + self.axnm[i]
            sel = self.axsel[i]
            try:
                ax = Gtool3Ax(path, sel).open()
            except:
                nm = ['z', 'y', 'x'][i] + '_NotFound'
                data = np.arange(self.shape[i])
                ax = xr.DataArray(
                    name = nm, 
                    data = data, 
                    dims = nm, 
                    coords = {nm:data}
                )
            axs.append(ax)
        self.dims = [ax.name for ax in axs]
        self.axs = axs
        return
    
    def read_data_chunk(self, i, j):
        # i番目のファイルのj番目のブロックのデータを読み出す
        st = (j+1)*self.blen - (self.datalen+DELIMLEN)
        chunk = self.read_chunk(i, st, self.datalen, self.ndtyp)
        return chunk
    
    def read_coef_chunk(self, i, j):
        # URY16フォーマットのとき、i番目のファイルのj番目のブロックの係数を読み出す
        st = (j+1)*self.blen - (3*DELIMLEN + self.coeflen + self.datalen)
        dtype = self.endian + 'f8'
        chunk = self.read_chunk(i, st, self.coeflen, dtype)
        return chunk
    
    def read_chunk(self, i, st, ln, dtype):
        chunk = da.from_delayed(
            dask.delayed(np.frombuffer(self.file(i, st, ln), dtype=dtype)), 
            shape=[ln/int(dtype[-1])], dtype=dtype
        )
        return chunk
    
    def file(self, i=0, st=0, ln=0):
        # i番目のファイルのstからlnまでを切り出す
        memmap = self.files[i]
        if st == ln:
            return memmap
        else:
            return memmap[st:st+ln]
        
class Gtool3Ax(Gtool3):

    def __init__(self, path, sel):
        self.sel = sel
        self.paths = [path]
        self.files = [np.memmap(path, 'S1', 'r')]
        self.parse_file()
        self.open()
        return

    def parse_file(self):
        self.interpret_delim()
        self.set_values1(self.read_header())
        self.set_values2()
        return

    def set_values2(self):
        # マスクの情報まで含めて変数を設定する
        self.datasize = self.size
        self.datalen = self.datasize*self.rlen
        # ブロック１つあたりの長さ
        if self.dfmt == 'URY16':
            self.blen = 6*DELIMLEN + HEADLEN + self.coeflen + self.datalen
        elif self.dfmt == 'MR4':
            self.blen = 3*DELIMLEN + HEADLEN + 2*12 + self.size//8 + self.datalen
        else:
            self.blen = 4*DELIMLEN + HEADLEN + self.datalen
        if self.filelen%self.blen != 0:
            # raise ValueError('BLEN is not correct!')
            print('filelen/blen = ', self.filelen/self.blen)
        self.nblocks = self.filelen//self.blen
        return

    def open(self):
        chunk = self.read_data_chunk(0,0)[self.sel]
        try:
            dim = DIMNAME[self.title]
        except:
            dim = self.title
        data = xr.DataArray(
            name = dim, 
            data = chunk, 
            dims = [dim], 
            coords = {dim:chunk}, 
            attrs = {'title':self.title, 'unit':self.unit, 'item':self.item}
        )
        data = data.where(data != self.miss)
        data = data.unstack()
        if self.dfmt == 'URY16':
            self.open_coef()
            data = self.offset + data*self.scale
        return data
    
class MultiFileGtool3(Gtool3):
    # 古いバージョンとの互換性確保のために残しています
    def mfopen(self):
        return super().open()