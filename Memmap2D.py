#!/usr/bin/env python


################################ system lib ##########################################
import os, sys
from pickle import TRUE
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import datetime as dt
import pandas as pd
import numpy as np
import akshare as ak
import csv
import quantstats as qs
import chinastock as cs


################################ private lib ##########################################

def __add_path__():
     if __name__ == '__main__':
         global CURRENT_PATH, HOME_PATH,SHARED_PATH
     CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
     HOME_PATH = os.path.abspath(os.path.join(CURRENT_PATH,'../')) # util
     if not (HOME_PATH in sys.path):
         print(f"ADDING HOME_PATH: {HOME_PATH}")
         sys.path.insert(0, HOME_PATH)
     SHARED_PATH = os.path.abspath(os.path.join(HOME_PATH,'shared')) # util
     if not (SHARED_PATH in sys.path):
         print(f"ADDING SHARED_PATH: {SHARED_PATH}")
         sys.path.insert(0, SHARED_PATH)
__add_path__()

from map.Map import *
##########################################################################


class M2D(Map):
    """ 
    Float data 2D map
    """
    HEADER_DTYPE = np.dtype('U33')
    SIZE_DTYPE = np.dtype('uint32')
    SYMBOL_DTYPE = np.dtype('U9')
    DATE_TYPE = np.dtype('uint32')
    DATA_DTYPE = np.dtype('float32')
    
    # EXT_NAME = 'f2d'
    
    @classmethod
    def __fp__(cls, path: str, section: str, mode: str):
        try:
            # 0. read dtype
            offset = 0
            fp = np.memmap(path, dtype=cls.HEADER_DTYPE, offset=offset, mode='r', shape = (4))
            dtypes = tuple(fp)
            del fp
            # 1. read size 
            offset = 4 * cls.HEADER_DTYPE.itemsize
            fp = np.memmap(path, dtype=cls.SIZE_DTYPE, offset=offset, mode='r', shape = (2))
            m , n  = tuple(fp)
            del fp
            if section == 0:
                offset = 4 * cls.HEADER_DTYPE.itemsize
                fp = np.memmap(path, dtype=cls.SIZE_DTYPE, offset=offset, mode=mode, shape = (2))
                return (fp, 2)
            elif section == 1:
                offset = 4 * cls.HEADER_DTYPE.itemsize+2 * cls.SIZE_DTYPE.itemsize
                fp = np.memmap(path, dtype=cls.SYMBOL_DTYPE, mode=mode, offset = offset,  shape = m)
                return (fp, m)
            elif section == 2:
                offset = 4 * cls.HEADER_DTYPE.itemsize+2 * cls.SIZE_DTYPE.itemsize + m * cls.SYMBOL_DTYPE.itemsize
                fp = np.memmap(path, dtype=cls.DATE_TYPE, mode=mode, offset = offset,  shape = n)
                return (fp, n)
            elif section == 3:
                offset = 4 * cls.HEADER_DTYPE.itemsize+2 * cls.SIZE_DTYPE.itemsize+\
                    m * cls.SYMBOL_DTYPE.itemsize + n * cls.DATE_TYPE.itemsize
                fp = np.memmap(path, dtype=cls.DATA_DTYPE, mode=mode, offset = offset,  shape = (m,n))
                return (fp, (m,n))
            
        except Exception as e:
            print(e)
            return None
        
    @classmethod
    def __save__(cls, df:pd.DataFrame, path: str):
        """index is symbols[str](will be truncated to U9), columns is dates[int], data point is float

        Args:
            df (pd.DataFrame): 
            
                       20090105  20090106  20090107  20090108  20090109  20090112  
            000001.SZ      9.71      10.3      9.99      9.60      9.85      9.86   
            000002.SZ      6.70       6.9      6.86      6.90      6.89      6.81   
            000004.SZ      3.69       3.8      3.99      4.19      4.35      4.26   

            path (str): path: alpha_name.jmap
        """
        try:
            assert df.index.is_object(), "index must be symbols[str] "
            assert df.columns.is_integer(), "column must be dates[int] "
            # sort dataframe by index:symbol and column:date
            df.sort_index(axis = 0, inplace =True)
            df.sort_index(axis = 1, inplace =True)

            # 0. save dtype
            offset = 0
            fp = np.memmap(path, dtype=cls.HEADER_DTYPE, offset=offset, mode='w+', shape = (4))
            fp[:] = [str(cls.SIZE_DTYPE), str(cls.SYMBOL_DTYPE),str(cls.DATE_TYPE),str(cls.DATA_DTYPE)]
            fp.flush()
            del fp
            
            # 1. save size 
            offset = 4 * cls.HEADER_DTYPE.itemsize 
            fp = np.memmap(path, dtype=cls.SIZE_DTYPE, offset=offset, mode='r+', shape = (2))
            fp[:] = df.shape
            fp.flush()
            del fp

            # 2. save symbol 
            offset = 4 * cls.HEADER_DTYPE.itemsize+2 * cls.SIZE_DTYPE.itemsize
            fp = np.memmap(path, dtype=cls.SYMBOL_DTYPE, mode='r+', offset = offset,  shape = df.shape[0])
            fp[:] = df.index.values.astype(cls.SYMBOL_DTYPE)
            fp.flush()
            del fp
            
            # 3. save date 
            offset = 4 * cls.HEADER_DTYPE.itemsize+2 * cls.SIZE_DTYPE.itemsize+  df.shape[0] * cls.SYMBOL_DTYPE.itemsize
            fp = np.memmap(path, dtype=cls.DATE_TYPE, mode='r+', offset = offset,  shape = df.shape[1])
            fp[:] = df.columns.values.astype(cls.DATE_TYPE)
            fp.flush()
            del fp
            
            # 4. save data 
            offset = 4 * cls.HEADER_DTYPE.itemsize+2 * cls.SIZE_DTYPE.itemsize+  \
                df.shape[0] * cls.SYMBOL_DTYPE.itemsize + df.shape[1] * cls.DATE_TYPE.itemsize
            fp = np.memmap(path, dtype=cls.DATA_DTYPE, mode='r+', offset = offset,  shape = df.shape)
            fp[:] = df.values.astype(cls.DATA_DTYPE)
            fp.flush()
            del fp
            return True
        except Exception as e:
            print(e)
            return False
        
    @classmethod
    def __update__(cls, df:pd.DataFrame, path: str):
        """only update number within the jmap. do not create new data point, will overrite all data
        Args:
            df (pd.DataFrame): 
            
                       20090105  20090106  20090107  20090108  20090109  20090112  
            000001.SZ      9.71      10.3      9.99      9.60      9.85      9.86   
            000002.SZ      6.70       6.9      6.86      6.90      6.89      6.81   
            000004.SZ      3.69       3.8      3.99      4.19      4.35      4.26   

            path (str): path: alpha_name.jmap
        """
        try:
            assert df.index.is_object(), "index must be symbols[str] "
            assert df.columns.is_integer(), "column must be dates[int] "
            
            # 0. save dtype
            offset = 0
            fp = np.memmap(path, dtype=cls.HEADER_DTYPE, offset=offset, mode='r', shape = (4))
            dtypes = tuple(fp)
            del fp
            
            # 1. read size 
            offset = 4 * cls.HEADER_DTYPE.itemsize 
            fp = np.memmap(path, dtype=cls.SIZE_DTYPE, offset=offset, mode='r', shape = (2))
            m , n  = tuple(fp)
            del fp

            # 2. read symbol 
            offset = 4 * cls.HEADER_DTYPE.itemsize +2 * cls.SIZE_DTYPE.itemsize
            fp = np.memmap(path, dtype=cls.SYMBOL_DTYPE, mode='r', offset = offset,  shape = m)
            symbols = np.array(fp)
            del fp
            
            # 3. read date 
            offset = 4 * cls.HEADER_DTYPE.itemsize +2 * cls.SIZE_DTYPE.itemsize + m * cls.SYMBOL_DTYPE.itemsize
            fp = np.memmap(path, dtype=cls.DATE_TYPE, mode='r', offset = offset,  shape = n)
            dates = np.array(fp)
            del fp
            
            # filter dataframe
            df_filtered = df.loc[df.index.isin(symbols), df.columns.isin(dates)]
            
            
            # 4. read data 
            offset = 4 * cls.HEADER_DTYPE.itemsize +2 * cls.SIZE_DTYPE.itemsize+ \
                m * cls.SYMBOL_DTYPE.itemsize + n * cls.DATE_TYPE.itemsize
            fp = np.memmap(path, dtype=cls.DATA_DTYPE, mode='r+', offset = offset,  shape = (m,n))

            filter_index = exact_searchsorted(symbols, df_filtered.index)
            assert np.all(filter_index>=0), "unknow index exists when update jmap"
            filter_columns = exact_searchsorted(dates, df_filtered.columns)
            assert np.all(filter_index>=0), "unknow column exists when update jmap"
            for i,j in zip(filter_columns,range(len(df_filtered.columns))):
                fp[filter_index,i] = df_filtered.values[filter_index,j].astype(cls.DATA_DTYPE)
            del fp
            return True
        except Exception as e:
            print(e)
            return False   
        
    @classmethod
    def __upsert__(cls, df:pd.DataFrame, path: str):
        """upsert data, write all new data into jmap. will create new data point

        Args:
            df (pd.DataFrame): 
            
                       20090105  20090106  20090107  20090108  20090109  20090112  
            000001.SZ      9.71      10.3      9.99      9.60      9.85      9.86   
            000002.SZ      6.70       6.9      6.86      6.90      6.89      6.81   
            000004.SZ      3.69       3.8      3.99      4.19      4.35      4.26   

            path (str): path: alpha_name.jmap
        """
        try:
            assert df.index.is_object(), "index must be symbols[str] "
            assert df.columns.is_integer(), "column must be dates[int] "
            df.columns = df.columns.astype('uint')
            old_df = cls.__read__(path)
            
            # new_index = list(df.index.values[exact_searchsorted(old_df.index.values, df.index.values)<0])
            # new_column = list(df.columns.values[exact_searchsorted(old_df.columns.values, df.columns.values)<0])
            new_index =  np.sort(pd.Series(np.concatenate([old_df.index.values, df.index.values])).unique())
            new_column = np.sort(pd.Series(np.concatenate([old_df.columns.values, df.columns.values])).unique())
            new_df = old_df.reindex(index = new_index, columns = new_column, copy = True)
            new_df.update(df, overwrite=True)
            return cls.__save__(new_df, path)
        except Exception as e:
            print(e)
            return False      
    
    @classmethod
    def __read__(cls, path: str ,sd = None, ed = None)-> pd.DataFrame: 
        
        """_summary_

        Args:
            df (pd.DataFrame): 
            
                       20090105  20090106  20090107  20090108  20090109  20090112  
            000001.SZ      9.71      10.3      9.99      9.60      9.85      9.86   
            000002.SZ      6.70       6.9      6.86      6.90      6.89      6.81   
            000004.SZ      3.69       3.8      3.99      4.19      4.35      4.26   

            path (str): path: alpha_name.jmap
        """
        try:
            # 0. save dtype
            offset = 0
            fp = np.memmap(path, dtype=cls.HEADER_DTYPE, offset=offset, mode='r', shape = (4))
            a = 1
            dtypes = tuple(fp)
            del fp
            
            # 1. read size 
            offset = 4 * cls.HEADER_DTYPE.itemsize
            fp = np.memmap(path, dtype=cls.SIZE_DTYPE, offset=offset, mode='r', shape = (2))
            m , n  = tuple(fp)
            del fp

            # 2. read symbol 
            offset = 4 * cls.HEADER_DTYPE.itemsize+2 * cls.SIZE_DTYPE.itemsize
            fp = np.memmap(path, dtype=cls.SYMBOL_DTYPE, mode='r', offset = offset,  shape = m)
            symbols = np.array(fp)
            del fp
           
            # 3. read date 
            #myList3 = []
            #myList4 = []
            #for i in range(100):
            #start = time.time()
            offset = 4 * cls.HEADER_DTYPE.itemsize+2 * cls.SIZE_DTYPE.itemsize + m * cls.SYMBOL_DTYPE.itemsize
            fp = np.memmap(path, dtype=cls.DATE_TYPE, mode='r', offset = offset,  shape = n)
            dates = np.array(fp)
            #end = time.time()
            #exeTime1 = end - start
            #myList3.append(exeTime1)
            del fp
                
            # 4. read data 
            #start = time.time()
            offset = 4 * cls.HEADER_DTYPE.itemsize+2 * cls.SIZE_DTYPE.itemsize+ \
                m * cls.SYMBOL_DTYPE.itemsize + n * cls.DATE_TYPE.itemsize
            fp = np.memmap(path, dtype=cls.DATA_DTYPE, mode='r', offset = offset,  shape = (m,n))
            sd_idx = 0
            ed_idx = None
            if not (sd is None): sd_idx = np.searchsorted(dates, sd, side ='left') 
            if not (ed is None): ed_idx = np.searchsorted(dates, ed, side ='right') 
            data = np.array(fp[:,sd_idx:ed_idx])
            del fp
            df = pd.DataFrame(data, index = symbols, columns = dates[sd_idx:ed_idx])
            return df
        except Exception as e:
            print(e)
            return None
        
    def __init__(self, path):
        self.path = path 
        
        # fp
        self._size_fp = self.__class__.__fp__(self.path, 0, 'r')[0]
        self._symbol_fp = self.__class__.__fp__(self.path, 1, 'r')[0]
        self._date_fp = self.__class__.__fp__(self.path, 2, 'r')[0]
        self._data_fp = self.__class__.__fp__(self.path, 3, 'r')[0]
        
        # size
        self.size =  tuple(self._size_fp)
        # symbol
        self.symbols =  np.array(self._symbol_fp)
        # date
        self.dates = np.array(self._data_fp)
        
        self.counter = 0
        
    def save(self,df):
        return self.__class__.__save__(df, self.path)  
    
    def update(self,df):
        return self.__class__.__update__(df, self.path)  
    
    def upsert(self,df):
        return self.__class__.__upsert__(df, self.path)  
    
    def read(self,sd = None, ed = None):
        return self.__class__.__read__(self.path, sd, ed)   
    
    def get_n(self, n=0):
        if n < self.size[1]:
            return np.array(self._data_fp[:,-n-1]) 
        else:
            return None
    
    def get(self):
        data = self.get_n(self.counter)
        self.counter += 1
        return data

##########################################################################    
""" np.bool """
class M2D_B(M2D):
    """ 
    bool data 2D map
    """
    DATA_DTYPE = np.dtype('bool')
    
    EXT_NAME = 'm2d_b' 

##########################################################################
""" np.int8 """
class M2D_I1(M2D):
    """ 
    int8 data 2D map
    """
    DATA_DTYPE = np.dtype('int8')
    
    EXT_NAME = 'm2d_i1'
    
""" np.int16 """
class M2D_I2(M2D):
    """ 
    int16 data 2D map
    """
    DATA_DTYPE = np.dtype('int16')
    
    EXT_NAME = 'm2d_i2'
    
""" np.int32 """
class M2D_I4(M2D):
    """ 
    int32 data 2D map
    """
    DATA_DTYPE = np.dtype('int32')
    
    EXT_NAME = 'm2d_i4'
    
""" np.int64 """
class M2D_I8(M2D):
    """ 
    int64 data 2D map
    """
    DATA_DTYPE = np.dtype('int64')
    
    EXT_NAME = 'm2d_i8'
    
##########################################################################
""" np.uint8 """
class M2D_U1(M2D):
    """ 
    uint8 data 2D map
    """
    DATA_DTYPE = np.dtype('uint8')
    
    EXT_NAME = 'm2d_u1'
    
""" np.uint16 """
class M2D_U2(M2D):
    """ 
    uint16 data 2D map
    """
    DATA_DTYPE = np.dtype('uint16')
    
    EXT_NAME = 'm2d_u2'
    
""" np.uint32 """
class M2D_U4(M2D):
    """ 
    uint32 data 2D map
    """
    DATA_DTYPE = np.dtype('uint32')
    
    EXT_NAME = 'm2d_u4'
    
""" np.uint64 """
class M2D_U8(M2D):
    """ 
    uint64 data 2D map
    """
    DATA_DTYPE = np.dtype('uint64')
    
    EXT_NAME = 'm2d_u8'
    
##########################################################################
""" np.float16 """
class M2D_F2(M2D):
    """ 
    float16 data 2D map
    """
    DATA_DTYPE = np.dtype('float16')
    
    EXT_NAME = 'm2d_f2'
    
""" np.float32 """
class M2D_F4(M2D):
    """ 
    float32 data 2D map
    """
    DATA_DTYPE = np.dtype('float32')
    
    EXT_NAME = '.m2d_f4'
   
""" np.float64 """
class M2D_F8(M2D):
    """ 
    float64 data 2D map
    """
    DATA_DTYPE = np.dtype('float64')
    
    EXT_NAME = 'm2d_f8' 

#def indStocks(vals: str ,ind: str, start = None, end = None): #Try using the pd.where() function in order to optimize runtime
#    #closing = pd.read_pickle("/Users/alexq/Downloads/memmap/close.pkl").T
#    #M2D_F4.__save__(closing, path = close) 
#    closing1 = M2D_F4.__read__(path = vals, sd = start, ed = end)

#    #indicator = pd.read_pickle("/Users/alexq/Downloads/memmap/indicator.pkl").T
#    #M2D_B.__save__(indicator, path = ind)
#    indicator1 = M2D_B.__read__(path = ind, sd = start, ed = end)

#    closing1_update = closing1.iloc[:,:].copy()
#    closing1_update.loc['test',:] = np.nan
#    closing1_update.loc[:, :] = closing1_update.loc[:, :].where(indicator1.loc[:, :], np.nan) 
#    M2D_F4.__update__(closing1_update, path = '/Users/alexq/Downloads/memmap/merge_file.m2d_f4')
#    closeR = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/merge_file.m2d_f4', sd = start, ed = end) 
#    return closeR
##########################################################################
# alias
M2D_Indicator = M2D_B
M2D_Alpha = M2D_F4


##########################################################################
if __name__ == '__main__': 
    close = pd.read_pickle("/Users/alexq/Downloads/memmap/indicator.pkl").T
    M2D_F4.__save__(close, path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4')
    close1 = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4', sd = 20200101, ed = 20201231)
    close = pd.read_pickle("/Users/alexq/Downloads/memmap/indicator.pkl").T
    M2D_B.__save__(close, path = '/Users/alexq/Downloads/memmap/indicator.m2d_b')
    close1 = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4', sd = 20200101, ed = 20201231)
    a=1

#if __name__ == '__main__': 
#    import timeit
#    #close = pd.read_pickle("/home/H01171/temp/close.pkl").T 
#    start = time.time()
#    close = pd.read_pickle("/Users/alexq/Downloads/memmap/close.pkl").T
#    print(time.time() - start)
#    close
#    #M2D_F4.__save__(close, path = '/home/H01171/temp/test_close.m2d_f4')
#    M2D_F4.__save__(close, path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4')
#    #close1 = M2D_F4.__read__(path = '/home/H01171/temp/test_close.m2d_f4', sd = 20200101, ed = 20201231)
#    start = time.time()
#    close1 = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4', sd = 20190101, ed = 20191231)
#    print(time.time()-start)
#    #print(close1)
#    a=1 
#if __name__ == '__main__': 
#    #close = pd.read_pickle("/home/H01171/temp/close.pkl").T 
#    close = pd.read_pickle("/Users/alexq/Downloads/memmap/close.pkl").T
#    close
#    #M2D_F4.__save__(close, path = '/home/H01171/temp/test_close.m2d_f4')
#    M2D_F4.__save__(close, path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4')
#    #close1 = M2D_F4.__read__(path = '/home/H01171/temp/test_close.m2d_f4', sd = 20200101, ed = 20201231)
#    close1 = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4', sd = 20200101, ed = 20200231)
#    a=1
    
#if __name__ == '__main__': 
#    #close = pd.read_pickle("/home/H01171/temp/close.pkl").T
#    close = pd.read_pickle("/Users/alexq/Downloads/memmap/close.pkl").T
#    close
#    #M2D_U1.__save__(close, path = '/home/H01171/temp/test_close.m2d_u1')
#    M2D_U1.__save__(close, path = '/Users/alexq/Downloads/memmap/test_close.m2d_u1')
#    #close1 = M2D_U1.__read__(path = '/home/H01171/temp/test_close.m2d_u1', sd = 20200101, ed = 20201231)
#    close1 = M2D_U1.__read__(path = '/Users/alexq/Downloads/memmap/test_close.m2d_u1', sd = 20200101, ed = 20201231)
#    a=1
    
    
#if __name__ == '__main__': 
#     close1 = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4', sd = 20200101, ed = 20201231)
#     close1_update = close1.iloc[:,[0,4]].copy()
#     close1_update.loc['000001.SZ',:] = np.nan
#     close1_update.loc[:,:] = 999
#     M2D_F4.__update__(close1_update, path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4')
#     close2 = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4', sd = 20200101, ed = 20201231)
#     print(close2)
#     a=1
    
#if __name__ == '__main__': 
#     close1 = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4', sd = 20200101, ed = 20201231)
#     close1_update = close1.iloc[:,[0,4]].copy()
#     close1_update.loc['test',:] = np.nan
#     close1_update.loc[:,:] = 999
#     M2D_F4.__upsert__(close1_update, path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4')
#     close2 = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4', sd = 20200101, ed = 20201231)
#     print(close2)
#     a=1

#if __name__ == '__main__': 
#    #close1 = M2D_F4('/home/H01171/temp/test_close.m2d_f4')
#    close1 = M2D_F4('/Users/alexq/Downloads/memmap/test_close.m2d_f4')
#    l = close1.get()
#    a=1

#if __name__ == '__main__':
#    #try using databases as parameters being read in???
#    close2 = indStocks(close = '/Users/alexq/Downloads/memmap/test_close.m2d_f4', ind = '/Users/alexq/Downloads/memmap/test_ind.m2d_B', start = 20200101, end = 20201231)
#    print(close2)
#    hi = ak.stock_zh_a_his(symbol = "02318", start_year = "2000", end_year = "2019")
#    print(hi)
#    dailyInd = ak.index_stock_cons_csindex("000300")
#    print(dailyInd)
#    a=1

def get_stock_prices(df:pd.DataFrame): #consider changing this into a batch api call and receive ##might be bad for macbook since only one core
    #set up and define empty files as open high low close and volume
    path = '/Users/alexq/Downloads/memmap/memmap2d/required_val/'
    file = ['open_val_df.m2d_f4','high_val_df.m2d_f4','low_val_df.m2d_f4','close_val_df.m2d_f4','volume_val_df.m2d_f4', 'amount_val_df.m2d_f4', 'hfq_high_val_df.m2d_f4', 'hfq_low_val_df.m2d_f4', 'hfq_close_val_df.m2d_f4']
    info = ['开盘','最高','最低','收盘','成交量', '成交额','最高','最低','收盘']
    open('/Users/alexq/Downloads/memmap/memmap2d/required_val/open_val_df.m2d_f4', 'rb')
    for i in range(9):
        M2D_F4.__save__(df, path = path+file[i])
        temp = M2D_F4.__read__(path = path+file[i])
        temp_update = temp.iloc[:,:].copy()
        temp_update.loc[:,:] = np.nan
        M2D_F4.__update__(temp_update, path = path+file[i])
    
    #each symbol needs to have all data inputted into the 5 files according to name
    #find start and end years if end year is == to start end year +1
    start_date = df.columns[0]
    end_date = df.columns[-1]
    #loop j inside row loop and assign each lookup to each file first

    #for row in df.columns: # row = dates col = symbols so use df.T
    for row in df.index.values: # row = dates col = symbols so use df.T
        use_row = row[0:6]
        try:
            dailyDF = ak.stock_zh_a_hist(symbol = use_row, start_date = start_date, end_date = end_date)
            dailyDF_hfq = ak.stock_zh_a_hist(symbol = use_row, start_date = start_date, end_date = end_date, adjust = 'hfq')
        except KeyError:
            dailyDF = np.nan
            dailyDF_hfq = np.nan
            continue
        #reformat dates yyyy-mm-dd -> yyyymmdd
        if dailyDF.empty:
            dailyDF = np.nan
            dailyDF_hfq = np.nan
            continue
        dailyDF['日期'] = pd.to_datetime(dailyDF.日期)
        dailyDF['日期'] = dailyDF['日期'].dt.strftime('%Y%m%d').astype(int)
        dailyDF.loc[:,'code'] = row
        dailyDF_hfq['日期'] = pd.to_datetime(dailyDF_hfq.日期)
        dailyDF_hfq['日期'] = dailyDF_hfq['日期'].dt.strftime('%Y%m%d').astype(int)
        dailyDF_hfq.loc[:,'code'] = row
        for j in range(9):
            if(j <= 5):
                pivotDF = dailyDF.pivot('code','日期',info[j])
                pivotDF.columns.name = None
                M2D_F4.__upsert__(pivotDF, path = path + file[j]) 
            else:
                pivotDF = dailyDF_hfq.pivot('code','日期',info[j])
                pivotDF.columns.name = None
                M2D_F4.__upsert__(pivotDF, path = path + file[j])  
    a = 1

#def get_stock_prices(df:pd.DataFrame): #consider changing this into a batch api call and receive ##might be bad for macbook since only one core
#    #set up and define empty files as open high low close and volume
#    path = '/Users/alexq/Downloads/memmap/memmap2d/required_val/'
#    file = ['open_val_df.m2d_f4','high_val_df.m2d_f4','low_val_df.m2d_f4','close_val_df.m2d_f4','volume_val_df.m2d_f4', 'amount_val_df.m2d_f4', 'hfq_high_val_df.m2d_f4', 'hfq_low_val_df.m2d_f4', 'hfq_close_val_df.m2d_f4']
#    info = ['开盘','最高','最低','收盘','成交量', '成交额','最高','最低','收盘']
#    open('/Users/alexq/Downloads/memmap/memmap2d/required_val/open_val_df.m2d_f4', 'rb')
#    for i in range(9):
#        M2D_F4.__save__(df, path = path+file[i])
#        temp = M2D_F4.__read__(path = path+file[i])
#        temp_update = temp.iloc[:,:].copy()
#        temp_update.loc[:,:] = np.nan
#        M2D_F4.__update__(temp_update, path = path+file[i])
    
#    #each symbol needs to have all data inputted into the 5 files according to name
#    #find start and end years if end year is == to start end year +1
#    start_date = df.columns[0]
#    end_date = df.columns[-1]
#    #loop j inside row loop and assign each lookup to each file first

#    #for row in df.columns: # row = dates col = symbols so use df.T
#    processes = []
#    processes_hfq = []
#    for row in df.index.values: # row = dates col = symbols so use df.T
#        use_row = df.index.values[0:6]
#        processA = multiprocessing.Process(target = ak.stock_zh_a_hist, args = ('symbol':use_row, 'start_date':start_date, 'end_date' : end_date))
#        processB = multiprocessing.Process(target = ak.stock_zh_a_hist, args = ('symbol':use_row, 'start_date':start_date, 'end_date' : end_date, 'adjust' : 'hfq' ))
#        processes.append(processA)
#        processes_hfq.append(processB)
#    for p in processes:
#        try:
#            dailyDF = ak.stock_zh_a_hist(symbol = use_row, start_date = start_date, end_date = end_date)
#            dailyDF_hfq = ak.stock_zh_a_hist(symbol = use_row, start_date = start_date, end_date = end_date, adjust = 'hfq')
#        except KeyError:
#            continue
#        #reformat dates yyyy-mm-dd -> yyyymmdd
#        if dailyDF.empty:
#            continue
#        dailyDF['日期'] = pd.to_datetime(dailyDF.日期)
#        dailyDF['日期'] = dailyDF['日期'].dt.strftime('%Y%m%d').astype(int)
#        dailyDF.loc[:,'code'] = row
#        dailyDF_hfq['日期'] = pd.to_datetime(dailyDF_hfq.日期)
#        dailyDF_hfq['日期'] = dailyDF_hfq['日期'].dt.strftime('%Y%m%d').astype(int)
#        dailyDF_hfq.loc[:,'code'] = row
#        for j in range(9):
#            if(j <= 5):
#                pivotDF = dailyDF.pivot('code','日期',info[j])
#                pivotDF.columns.name = None
#                M2D_F4.__upsert__(pivotDF, path = path + file[j]) 
#            else:
#                pivotDF = dailyDF_hfq.pivot('code','日期',info[j])
#                pivotDF.columns.name = None
#                M2D_F4.__upsert__(pivotDF, path = path + file[j])  
#    a = 1

def get_indicator_values(df:pd.DataFrame): #when this works for all keys also convert this to a batching process to save runtime
    path = '/Users/alexq/Downloads/memmap/memmap2d/required_val/indicator.m2d_b'
    M2D_B.__save__(df, path = path)
    temp = M2D_B.__read__(path = path)
    temp_update = temp.iloc[:,:].copy()
    temp_update.loc[:,:] = False
    M2D_B.__update__(temp_update, path = path)
    index_df = M2D_B.__read__(path = path)
    with open('AINDEXMEMBERS_202306290954.csv', newline = '', mode = 'r') as file:
        csvFile = csv.reader(file)
        list = []
        for lines in csvFile:
            list.append(lines)
        for row in index_df.T:
            i=0
            for i in range(len(list)):
                if row == list[i][0]:
                    for col in index_df:
                        if col >= int(list[i][1]):
                            if list[i][2] == '' or col <= int(list[i][2]):
                                index_df.loc[row,col] = True
    M2D_Indicator.__update__(index_df, path = path)

def calc_valuestd(path: str, sd, ed): #use this one optimal run time
    JD = 20 #optimal value as found by the study
    #for safety reasons always look 50 days before the start date to account for holidays and breaks and weekends
    new_sd = dt.datetime.strptime(str(sd),'%Y%m%d')
    new_sd = new_sd + dt.timedelta(days = -50)
    new_sd = int(new_sd.strftime('%Y%m%d'))

    #read the file we want with 50 days in mind
    #df = M2D_F4.__read__(path = path, sd = new_sd, ed= ed)
    df = M2D_F4.__read__(path = path + 'index_close_df.m2d_f4', sd = new_sd, ed = ed)
    a = 1
    #after reading file with the adjusted start date we want to truncate such that the file has exactly 20 dates of data beforehand
    #df1 = M2D_F4.__read__(path = path, sd = sd, ed= ed) 
    df1 = M2D_F4.__read__(path = path + '/index_volume_df.m2d_f4', sd = sd, ed= ed)

    #df1 is the dataframe we are modifying and ultimately saving as the new file
    col_idx = df.columns.get_loc(df1.columns[0])
    new_sd = df.columns[col_idx-JD] #if JD of 20 means 21 values then change 19 to 20
    #df = M2D_F4.__read__(path = path, sd = new_sd, ed= ed)
    df = M2D_F4.__read__(path = path + '/index_volume_df.m2d_f4', sd = new_sd, ed = ed)

    #calculate the value and save to a properly sized table that has the original start date and not the adjusted
    colidx = 0
    for col in df1:
        if col == np.nan:
            colidx = colidx+1
            continue
        standard = df.iloc[:,colidx:colidx+JD].std(axis = 1, numeric_only= True)
        df1.loc[:,col] = 1/standard
        colidx = colidx+1
        #for col in df:
    with open(os.path.join('/Users/alexq/Downloads/memmap/memmap2d/calculated_idx_val', 'idxvalstd.m2d_f4'), 'w') as fp:
        pass
    df1.replace([np.inf, -np.inf], 0, inplace=True)
    M2D_F4.__save__(df1, path = '/Users/alexq/Downloads/memmap/memmap2d/calculated_idx_val/idxvalstd.m2d_f4')


def calc_longshort(path: str, sd, ed): #this is most annoying have to change definition of prevclose to literally previous close
    #set up the dataframe like in the previous one
    #first calculate for closing price then preclose price

    JD_close = 90 #optimal value as found by the study for close benchmark
    JD_open = 5
    #for safety reasons always look 50 days before the start date to account for holidays and breaks and weekends
    new_sd = dt.datetime.strptime(str(sd),'%Y%m%d')
    new_sd = new_sd + dt.timedelta(days = -220)
    new_sd = int(new_sd.strftime('%Y%m%d'))

    #read the file we want with 50 days in mind
    #df = M2D_F4.__read__(path = path, sd = new_sd, ed= ed)
    high = M2D_F4.__read__(path = path + '/index_high_df.m2d_f4', sd = new_sd, ed = ed)
    #after reading file with the adjusted start date we want to truncate such that the file has exactly 90 dates of data beforehand
    #df1 = M2D_F4.__read__(path = path, sd = sd, ed= ed) 
    df1 = M2D_F4.__read__(path = path + '/index_volume_df.m2d_f4', sd = sd, ed= ed)

    #df1 is the dataframe we are modifying and ultimately saving as the new file
    col_idx = high.columns.get_loc(df1.columns[0])
    new_sd_close = high.columns[col_idx-JD_close]
    new_sd_open =  high.columns[col_idx-JD_open]
    #df = M2D_F4.__read__(path = path, sd = new_sd, ed= ed)
    high = M2D_F4.__read__(path = path + '/index_high_df.m2d_f4', sd = new_sd_open, ed = ed)
    low = M2D_F4.__read__(path = path + '/index_low_df.m2d_f4', sd = new_sd_open, ed = ed) 
    opening = M2D_F4.__read__(path = path + '/index_open_df.m2d_f4', sd = new_sd_open, ed = ed)
    closing = M2D_F4.__read__(path = path + '/index_close_df.m2d_f4', sd = new_sd_open, ed = ed)
    high1 = M2D_F4.__read__(path = path + '/index_high_df.m2d_f4', sd = new_sd_close, ed = ed)
    low1 = M2D_F4.__read__(path = path + '/index_low_df.m2d_f4', sd = new_sd_close, ed = ed) 
    opening1 = M2D_F4.__read__(path = path + '/index_open_df.m2d_f4', sd = new_sd_close, ed = ed)
    closing1 = M2D_F4.__read__(path = path + '/index_close_df.m2d_f4', sd = new_sd_close, ed = ed)
    #high_hfq = M2D_F4.__read__(path = path + '/hfq_high_val_df.m2d_f4', sd = new_sd_close, ed = ed)
    #low_hfq = M2D_F4.__read__(path = path + '/hfq_low_val_df.m2d_f4', sd = new_sd_close, ed = ed) 
    #closing_hfq = M2D_F4.__read__(path = path + '/hfq_close_val_df.m2d_f4', sd = new_sd_close, ed = ed)
    #calculate the long-short(closing + long-short(preclosing))
    colidx = 0
    for col in df1:
        if col == np.nan:
            colidx = colidx + 1 
            continue
        vals = ((high.iloc[:, colidx:colidx+JD_open] - opening.iloc[:,colidx:colidx+JD_open])/((1+ opening.iloc[:,colidx:colidx+JD_open]-low.iloc[:,colidx:colidx+JD_open])))
        vals = vals.replace([np.inf, -np.inf], 0, inplace=False).sum(axis = 1, numeric_only = True)
        #vals2 should use the hfq values
        vals2 = ((high1.iloc[:, colidx-1:colidx+JD_close-1] - closing1.iloc[:,colidx-1:colidx-1+JD_close])/(1+(closing1.iloc[:,colidx-1:colidx+JD_close-1]-low1.iloc[:,colidx-1:colidx+JD_close-1])))
        vals2 = vals2.replace([np.inf, -np.inf], np.nan, inplace=False).sum(axis = 1, numeric_only = True)
        df1.loc[:,col] = vals + vals2
        colidx = colidx+1
    with open(os.path.join('/Users/alexq/Downloads/memmap/memmap2d/calculated_idx_val', 'idxlongshort.m2d_f4'), 'w') as fp:
        pass
    df1.replace([np.inf, -np.inf], 0, inplace=True)
    M2D_F4.__save__(df1, path = '/Users/alexq/Downloads/memmap/memmap2d/calculated_idx_val/idxlongshort.m2d_f4')

def calc_trends_up(path: str, sd, ed):
    JD = 40 #optimal value as found by the study
    TB = 100 #traceback in order to get the value
    #for safety reasons always look 170 days before the start date to account for holidays and breaks and weekends
    new_sd = dt.datetime.strptime(str(sd),'%Y%m%d')
    new_sd = new_sd + dt.timedelta(days = -170)
    new_sd = int(new_sd.strftime('%Y%m%d'))

    #read the file we want with 100 days in mind
    #df = M2D_F4.__read__(path = path, sd = new_sd, ed= ed)
    df1 = M2D_F4.__read__(path = path + '/index_close_df.m2d_f4', sd = new_sd, ed = ed)
    #after reading file with the adjusted start date we want to truncate such that the file has exactly 40 dates of data beforehand
    #df1 = M2D_F4.__read__(path = path, sd = sd, ed= ed) 
    close = M2D_F4.__read__(path = path + '/index_close_df.m2d_f4', sd = sd, ed= ed)

    #close is the dataframe we are modifying and ultimately saving as the new file
    col_idx = df1.columns.get_loc(close.columns[0])
    new_sd = df1.columns[col_idx-TB] #if TB of 100 means 101 values then change 99 to 100
    high = M2D_F4.__read__(path = path + '/index_high_df.m2d_f4', sd = new_sd, ed = ed)
    nandf = M2D_F4.__read__(path = path + '/index_close_df.m2d_f4', sd = sd, ed = ed) #nan dataframe
    df1 = M2D_F4.__read__(path = path + '/index_close_df.m2d_f4', sd = new_sd, ed= ed)
    nandf.loc[:,:] = np.nan
    #close = close.where(df1.loc[:,col] > df1.loc[:,df1.columns[df1.columns.get_loc(col)-JD+1]], DataFrame = nandf, axis = 1)
    for col in close:
        if col == np.nan:
            col_idx = col_idx+1
            continue
        close.loc[:,col] = close.loc[:,col].where(df1.loc[:,col] > df1.loc[:,df1.columns[df1.columns.get_loc(col)-JD]], other = nandf.loc[:,col])#column by column solution
        divisor = high.iloc[:,col_idx-TB:col_idx].max(axis = 1, numeric_only = True)
        close.loc[:,col] = close.loc[:,col]/divisor
        col_idx = col_idx+1 
    with open(os.path.join('/Users/alexq/Downloads/memmap/memmap2d/calculated_idx_val', 'idxTrendsUp.m2d_f4'), 'w') as fp:
        pass
    close.replace([np.inf, -np.inf], 0, inplace=True)
    M2D_F4.__save__(close, path = '/Users/alexq/Downloads/memmap/memmap2d/calculated_idx_val/idxTrendsUp.m2d_f4')


def calc_trends_down(path: str, sd, ed):
    JD = 40 #optimal value as found by the study
    TB = 500 #traceback in order to get the value
    #for safety reasons always look 800 days before the start date to account for holidays and breaks and weekends
    new_sd = dt.datetime.strptime(str(sd),'%Y%m%d')
    new_sd = new_sd + dt.timedelta(days = -800)
    new_sd = int(new_sd.strftime('%Y%m%d'))

    #read the file we want with 100 days in mind
    #df = M2D_F4.__read__(path = path, sd = new_sd, ed= ed)
    df1 = M2D_F4.__read__(path = path + '/index_close_df.m2d_f4', sd = new_sd, ed = ed)
    #after reading file with the adjusted start date we want to truncate such that the file has exactly 40 dates of data beforehand
    #df1 = M2D_F4.__read__(path = path, sd = sd, ed= ed) 
    close = M2D_F4.__read__(path = path + '/index_close_df.m2d_f4', sd = sd, ed= ed)

    #close is the dataframe we are modifying and ultimately saving as the new file
    col_idx = df1.columns.get_loc(close.columns[0])
    new_sd = df1.columns[col_idx-TB] #if TB of 100 means 101 values then change 99 to 100
    low = M2D_F4.__read__(path = path + '/index_low_df.m2d_f4', sd = new_sd, ed = ed)
    nandf = M2D_F4.__read__(path = path + '/index_close_df.m2d_f4', sd = sd, ed = ed) #nan dataframe
    df1 = M2D_F4.__read__(path = path + '/index_close_df.m2d_f4', sd = new_sd, ed= ed)
    nandf.loc[:,:] = np.nan
    #close = close.where(close.loc[:,:] < df1.loc[:,df1.columns[df1.columns.get_loc(col)-JD+1]], other = nandf, axis = 1) #try to calculate entire thing at once to reduce runtime
    for col in close:
        if col == np.nan:
            col_idx = col_idx+1
            continue
        close.loc[:,col] = close.loc[:,col].where(df1.loc[:,col] < df1.loc[:,df1.columns[df1.columns.get_loc(col)-JD]], other = nandf.loc[:,col])#column by column solution
        divisor = low.iloc[:,col_idx-TB:col_idx].min(axis = 1, numeric_only = True)
        close.loc[:,col] = close.loc[:,col]/divisor
        col_idx = col_idx+1 
    with open(os.path.join('/Users/alexq/Downloads/memmap/memmap2d/calculated_idx_val', 'idxTrendsDown.m2d_f4'), 'w') as fp:
        pass
    close.replace([np.inf, -np.inf], 0, inplace=True)
    M2D_F4.__save__(close, path = '/Users/alexq/Downloads/memmap/memmap2d/calculated_idx_val/idxTrendsDown.m2d_f4')

def get_Complex_Factor(path: str, sd, ed):
    valstd_df = M2D_F4.__read__(path = path+'/idxvalstd.m2d_f4', sd = sd, ed = ed)
    trends_up_df = M2D_F4.__read__(path = path+'/idxTrendsUp.m2d_f4', sd = sd, ed = ed)
    trends_down_df = M2D_F4.__read__(path = path+'/idxTrendsDown.m2d_f4', sd = sd, ed = ed)
    longshort_df = M2D_F4.__read__(path = path+'/idxlongshort.m2d_f4', sd = sd, ed = ed)
    complex_df = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/memmap2d/index_val/index_amount_df.m2d_f4', sd = sd, ed = ed)
    with open(os.path.join('/Users/alexq/Downloads/memmap/memmap2d/calculated_idx_val', 'idxalpha_val.m2d_f4'), 'w') as fp:
        pass
    #for col in complex_df.columns: #goal only add trendsup or trendsdown depending if nan exists in the value spot
    #    complex_df.loc[:,col] = valstd_df.loc[:,col].rank( na_option = 'keep', numeric_only = True, pct = True).fillna(0)\
    #        + trends_up_df.loc[:,col].rank(numeric_only = True, na_option = 'keep', pct = True).fillna(0)\
    #              + trends_down_df.loc[:,col].rank( numeric_only = True, na_option = 'keep',pct = True).fillna(0)\
    #                  + longshort_df.loc[:,col].rank( numeric_only = True, na_option = 'keep', pct = True).fillna(0) #think about what to fill na with maybe an extremely large number instead of 0
    #for col in complex_df.columns: #goal only add trendsup or trendsdown depending if nan exists in the value spot
    #    complex_df.loc[:,col] = valstd_df.loc[:,col].rank(ascending = False, na_option = 'keep', numeric_only = True, pct = True).fillna(1)\
    #        + trends_up_df.loc[:,col].rank(ascending = False, numeric_only = True, na_option = 'keep', pct = True).fillna(1)\
    #              + trends_down_df.loc[:,col].rank(ascending = False, numeric_only = True, na_option = 'keep',pct = True).fillna(1)\
    #                  + longshort_df.loc[:,col].rank(ascending = False, numeric_only = True, na_option = 'keep', pct = True).fillna(1)
    #complex_df = complex_df.shift(1, axis = 1)

    for col in complex_df.columns: #goal only add trendsup or trendsdown depending if nan exists in the value spot
        alpha1 = valstd_df.loc[:,col].rank(ascending = False, na_option = 'keep', numeric_only = True, pct = True)\
        +  trends_up_df.loc[:,col].rank(ascending = False, numeric_only = True, na_option = 'keep', pct = True)\
            + longshort_df.loc[:,col].rank(ascending = False, numeric_only = True, na_option = 'keep', pct = True)
    
        alpha2 = valstd_df.loc[:,col].rank(ascending = False, na_option = 'keep', numeric_only = True, pct = True)\
            + trends_down_df.loc[:,col].rank(ascending = False, numeric_only = True, na_option = 'keep',pct = True)\
                + longshort_df.loc[:,col].rank(ascending = False, numeric_only = True, na_option = 'keep', pct = True)
        
        complex_df.loc[:,col] = alpha1.where(alpha2.isna(), alpha2)
    complex_df = complex_df.shift(1, axis = 1)
    
    M2D_Alpha.__save__(complex_df, path = '/Users/alexq/Downloads/memmap/memmap2d/calculated_idx_val/idxalpha_val.m2d_f4')

#if __name__ == '__main__':
#    #print(close1)
#    path = '/Users/alexq/Downloads/memmap/memmap2d/required_val/'
#    file = ['open_val_df.m2d_f4','high_val_df.m2d_f4','low_val_df.m2d_f4','close_val_df.m2d_f4','turnover_val_df.m2d_f4'] 
#    get_indicator_values(close1)
#    get_stock_prices(close1)
#    print(M2D_B.__read__(path = '/Users/alexq/Downloads/memmap/required_val/indicator.m2d_b'))
#    for i in range(len(file)):
#        print(M2D_F4).__read__(path = path+file[i])

if __name__ == '__main__':
    #close1 = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4', sd = 20150101, ed = 20201231)
    #print(close1.columns[500])
    #get_stock_prices(close1)
    #get_indicator_values(close1)
    calc_valuestd(sd = 20090123, ed = 20230531 , path = '/Users/alexq/Downloads/memmap/memmap2d/index_val/')
    calc_longshort(sd = 20090123, ed = 20230531, path = '/Users/alexq/Downloads/memmap/memmap2d/index_val/')
    calc_trends_down(sd = 20090123, ed = 20230531, path = '/Users/alexq/Downloads/memmap/memmap2d/index_val/')
    calc_trends_up(sd = 20090123, ed = 20230531, path = '/Users/alexq/Downloads/memmap/memmap2d/index_val/')
    a=1
    get_Complex_Factor(path = '/Users/alexq/Downloads/memmap/memmap2d/calculated_idx_val', sd = 20090123, ed = 20230531)
    a=1

#if __name__ == '__main__':
#  close = pd.read_pickle("/Users/alexq/Downloads/memmap/close.pkl").T
#    #print(time.time() - start)
#  close
##    #M2D_F4.__save__(close, path = '/home/H01171/temp/test_close.m2d_f4')
#  M2D_F4.__save__(close, path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4')
##    #close1 = M2D_F4.__read__(path = '/home/H01171/temp/test_close.m2d_f4', sd = 20200101, ed = 20201231)
##    #start = time.time()
#  close1 = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4', sd = 20200101, ed = 20201231)
#    #print(time.time() - start)
  
#  ind = pd.read_pickle("/Users/alexq/Downloads/memmap/indicator.pkl").T
#  ind
#  M2D_B.__save__(ind, path = '/Users/alexq/Downloads/memmap/test_ind.m2d_B')
#  ind1 = M2D_B.__read__(path = '/Users/alexq/Downloads/memmap/test_ind.m2d_B', sd = 20200101, ed = 20201231)
#    #print(time.time() - start)

#  #close1[ind1 == False] = np.nan
#  start = time.time()
#  close1_update = close1.iloc[:,:].copy()
#  close1_update.loc['test',:] = np.nan
#  close1_update.loc[:, :] = close1_update.loc[:, :].where(ind1.loc[:, :], np.nan)
#  M2D_F4.__update__(close1_update, path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4')
#  close2 = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/test_close.m2d_f4', sd = 20200101, ed = 20201231)
#  print(time.time() - start)
#  print(close2)


#IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Next steps are using the new install we are going to perform this comparison by downloading and formatting
#online stock data with 5 sets of data each
#We will then download and reformat a data set of indicator data sets
#We will then use a formula to compute the alpha values
#We will use alpha values to then determine top 10 stocks to buy


#complex alpha factor = rank[Trends up] + rank[Trends down] + rank[Value std] + rank[Long Short]
#how to treat ranks for Trends up and Trends down? Treat nan as? equals case is 0 or nan?

#############################################################################
#BackTester we are coding our own framework. Create a class init with parameters such as start date, end date, etc...
#have self.value and self.alpha and call trade(self)

#backtest(self) for all dates in start date to end date, we want to call the trade function which will loop from start date to end date

#Two periods or parts to the code, the morning and the close time
#morning: We want to convert alpha into weight so set value to some n, this will be the starting amount we put into our investments
#to get weight we need to unrankify alpha so we want to cancel out the rank to get a percentage so first determine how many symbols we are
#testing, top 6 top 10 .... any number then divide the ranked alpha of each of those stocks by the total number of alpha so we get a percent
# weight of each stock to distribute. Multiply our each weight by the value to get the target amount of money spent per stock
#Then to find position/#shares, divide target by the open price
#Do not mutate dataframes, save the weight, target, and position data frames maybe a list of dataframes? or dictionary with the date as key?

#close time: at night we want to reverse the morning process to find our new value, so take the position and multiply it by close price
#to get the value at close, then we want to save that value as well as another variable to find net profit or gain
#net change will be value(close) - value(open) 

class backtester:


    def __init__(self, sd, ed, value, lib_path:str, top_num:int, freq:int):
        self.value = value
        self.tracker = []

        #self.alpha = M2D_F4.__read__(path = lib_path + 'calculated_val/alpha_val.m2d_f4', sd=sd, ed=ed)

        self.alpha = M2D_F4.__read__(path = lib_path + 'calculated_idx_val/idxalpha_val.m2d_f4', sd=sd, ed=ed) #index

        #self.alpha = pd.read_pickle('/Users/alexq/Downloads/memmap/test_pic_alpha-4.pkl')
        #self.alpha.index = self.alpha.index.droplevel('alpha_name')
        #M2D_F4.__save__(self.alpha,path = '/Users/alexq/Downloads/memmap/test_pic_alpha-4.m2d_f4')
        #self.alpha = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/test_pic_alpha-4.m2d_f4', sd = sd, ed=ed)

        #self.alpha = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/sample_alpha.m2d_f4', sd = sd, ed = ed)
        #self.alpha = self.alpha.loc["000001.SZ":"688389.SH" ,sd:ed]

        self.top_num = top_num
        self.net_val = 0
        #self.open = M2D_F4.__read__(path = lib_path + 'required_val/open_val_df.m2d_f4', sd=sd, ed=ed)
        #self.close = M2D_F4.__read__(path = lib_path + 'required_val/close_val_df.m2d_f4', sd=sd, ed=ed)

        #index
        self.open = M2D_F4.__read__(path = lib_path + 'index_val/index_open_df.m2d_f4', sd=sd, ed=ed)
        self.close = M2D_F4.__read__(path = lib_path + 'index_val/index_close_df.m2d_f4', sd=sd, ed=ed)


        self.daily_trade_vals = {}
        self.cummulative_pnl = pd.DataFrame(index = self.alpha.columns, columns = None)
        self.freq = freq
        self.portfolio = {}
        #self.portfolio1 = pd.DataFrame()
        self.port_val = 0
        self.prev = 0
    
    def price_check_buy(self, symbols, date): #symbols should be top stocks do not add more stocks use after putting top_stocks into list
        if len(self.daily_trade_vals) == 0:
            for symbol in symbols.index:
                open_ref = self.open.loc[symbol,date]
                if np.isnan(open_ref):
                    symbols.loc[symbol] = 0
            return
        for symbol in symbols.index: #might not make this a function just perform within the trade function might be easier to code
            prev_close = self.close.loc[symbol, self.close.columns[self.close.columns.get_loc(date)-1]]
            #lowerbound = prev_close * 0.9 # cannot sell
            upperbound = prev_close * 1.1 # cannot buy
            open_ref = self.open.loc[symbol,date]
            if open_ref >= upperbound or np.isnan(open_ref):#some logic
                symbols.loc[symbol] = 0

    def price_check_sell(self, portfolio, date, sell_price): #symbols should be top stocks do not add more stocks use after putting top_stocks into list
        if len(self.daily_trade_vals) == 1:
            #sell all need to adjust for np.nan case do that last
            self.value = sell_price.sum()
            self.port_val = 0
            portfolio = portfolio.clear()
            return
        else:
            proxy = portfolio.copy()
            for symbol in proxy: 
                prev_close = self.close.loc[symbol, self.close.columns[self.close.columns.get_loc(date)-1]]
                lowerbound = prev_close * 0.9 
                open_ref = self.open.loc[symbol,date]
                if open_ref >= lowerbound and not np.isnan(self.close.loc[symbol,date]):
                    self.value = self.value + (self.open.loc[symbol,date] * portfolio[symbol])
                    self.port_val = self.port_val - (self.open.loc[symbol,date] * portfolio[symbol])
                    portfolio.pop(symbol)

    

    #def buy(self, date):
    #    #initializing tracker to keep up with trades
    #    self.tracker = []
    #    if len(self.daily_trade_vals) == 0:
    #        self.prev = self.port_val + self.value
    #    #finding top n stocks to check and buy
    #    stock_buy = self.alpha.loc[:,date].rank(ascending = True, na_option = 'keep', numeric_only = True, pct = True).sort_values(ascending = False, na_position = 'last').iloc[0:self.top_num]
    #    #stock_buy = self.alpha.loc[:,date].rank(ascending = False, na_option = 'keep', numeric_only = True, pct = True).sort_values(ascending = False, na_position = 'last').iloc[0:self.top_num]
        
    #    #ranks = (self.alpha.loc[:,date].dropna().rank(ascending = False, na_option = 'keep', numeric_only = True, pct = True) -0.5).sort_values(ascending = False, na_position = 'last')
    #    #stock_pos = ranks.iloc[0:25]
    #    #stock_neg = ranks.dropna().iloc[-25:]
    #    #stock_buy = stock_pos.append(stock_neg).abs()

    #    self.price_check_buy(symbols = stock_buy, date = date) #check before calculating weight
    #    self.tracker.append({"top_stocks":stock_buy})

    #    #calculating and recording the weight
    #    weight = stock_buy/stock_buy.sum() 
    #    self.tracker.append({"weight":weight})

    #    #calculating and recording the amount invested per stock
    #    target = weight*self.value
    #    self.tracker.append({"target":target})

    #    #determining how many shares of each stock, recording it in both our daily tracker and portfolio
    #    assert self.value >= 0
    #    position = target/self.open.loc[target.index, date]
    #    position = position.dropna(axis=0, how='any', inplace=False)
    #    temp = {}
    #    pos_dict = position.T.to_dict()

    #    sum = 0
    #    for symbol in self.portfolio: 
    #        if np.isnan(self.close.loc[symbol,date]):
    #            sum = sum + self.portfolio[symbol] * self.close.loc[symbol,:date].fillna(method='ffill').iloc[-1] #repetitive with the part in sell


    #    for key in self.portfolio:
    #        temp[key] = self.portfolio[key] + pos_dict.get(key, 0) 
    #    for key in pos_dict:
    #        if key not in temp:
    #            temp[key] = pos_dict[key]
    #    self.portfolio = temp
    #    #self.portfolio.update(position.T.to_dict()) #need to avoid mutating existing stock
    #    self.tracker.append({"position":position})
    #    #recording and finding out the total monetary value of my stocks
    #    close_val = position*self.close.loc[position.index,date] #sell value of all my stocks
    #    self.port_val = sum + self.port_val
    #    self.port_val = close_val.sum() + self.port_val
    #    self.tracker.append({"close":close_val})
    #     #using portfolio value - the net_value for the temporary delta and then changing the net_value
    #    self.value = 0
    #    delta_value = self.port_val - self.prev 
    #    self.net_val = self.net_val + delta_value
    #    self.cummulative_pnl.loc[date,'pnl'] = self.net_val #set the net_val into our P&L
    #    self.prev = self.port_val
    #    self.tracker.append({"delta_val":delta_value})
    #    self.tracker.append({"net_val":self.net_val})
    #    try:
    #        self.daily_trade_vals[date].append(self.tracker)
    #    except KeyError:
    #        self.daily_trade_vals[date] = self.tracker
        
    #    #pnl end of day
    
    def buy(self, date):
        #initializing tracker to keep up with trades
        self.tracker = []
        if len(self.daily_trade_vals) == 0:
            self.prev = self.port_val + self.value

        ranks = (self.alpha.loc[:,date].dropna().rank(pct = True) -0.5).sort_values(ascending = False, na_position = 'last')
        stock_top = ranks.iloc[0:self.top_num]
        stock_bot = ranks.dropna().iloc[-self.top_num:].abs()

        self.price_check_buy(symbols = stock_top, date = date) #check before calculating weight
        self.price_check_buy(symbols = stock_bot, date = date)
        self.tracker.append({"top_stocks":stock_top.append(stock_bot)})

        #calculating and recording the weight
        weight_top = stock_top/stock_top.sum() 
        weight_bot = stock_bot/stock_bot.sum()
        self.tracker.append({"weight":weight_top.append(weight_bot)})
        #ref_check = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/sample_weight.m2d_f4', sd = 20161230, ed = 20200102).loc[:,date]
        #for row in ref_check.index:
        #    if ref_check.loc[row] != 0.0 and not np.isnan(ref_check.loc[row]):
        #        print(row)
        #        print(ref_check.loc[row])

        #calculating and recording the amount invested per stock
        target_top = weight_top * self.value * 0.5
        target_bot = weight_bot * self.value * 0.5
        if weight_top.sum() <= 0.0 and not weight_bot.sum() == 0.0 :
            target_bot = target_bot * 2
        elif weight_bot.sum() <= 0.0 and not weight_top.sum() == 0.0 :
            target_top = target_top * 2
        elif weight_bot.sum() == 0 and weight_top.sum() == 0:
            return
        target = target_top.append(target_bot)
        #target = target_top
        self.tracker.append({"target":target})

        #determining how many shares of each stock, recording it in both our daily tracker and portfolio
        assert self.value >= 0
        position = target/self.open.loc[target.index, date]
        position = position.dropna(axis=0, how='any', inplace=False)
        temp = {}
        pos_dict = position.T.to_dict()

        sum = 0
        for symbol in self.portfolio: 
            if np.isnan(self.close.loc[symbol,date]):
                sum = sum + self.portfolio[symbol] * self.close.loc[symbol,:date].fillna(method='ffill').iloc[-1] #repetitive with the part in sell


        for key in self.portfolio:
            temp[key] = self.portfolio[key] + pos_dict.get(key, 0) 
        for key in pos_dict:
            if key not in temp:
                temp[key] = pos_dict[key]
        self.portfolio = temp
        #self.portfolio.update(position.T.to_dict()) #need to avoid mutating existing stock
        self.tracker.append({"position":position})
        #recording and finding out the total monetary value of my stocks
        close_val = position*self.close.loc[position.index,date] #sell value of all my stocks
        self.port_val = sum + self.port_val
        self.port_val = close_val.sum() + self.port_val
        self.tracker.append({"close":close_val})
         #using portfolio value - the net_value for the temporary delta and then changing the net_value
        self.value = 0
        delta_value = self.port_val - self.prev 
        self.net_val = self.net_val + delta_value
        self.cummulative_pnl.loc[date,'pnl'] = self.net_val #set the net_val into our P&L
        self.prev = self.port_val
        self.tracker.append({"delta_val":delta_value})
        self.tracker.append({"net_val":self.net_val})
        try:
            self.daily_trade_vals[date].append(self.tracker)
        except KeyError:
            self.daily_trade_vals[date] = self.tracker
        
        #pnl end of day
        
    #def sell(self,date):
    #    temp = self.open.loc[self.portfolio.keys(),date]
    #    for row in self.open.loc[self.portfolio.keys(),date].index: #calculate isnan values here instead?
    #        temp.loc[row] = temp.loc[row]*self.portfolio.get(row)
    #    open_val = temp.loc[self.portfolio.keys()]
    #    self.port_val = open_val.sum()
    #    #price check and sell stocks
    #    self.price_check_sell(portfolio = self.portfolio, sell_price = open_val, date = date)
    #    try:
    #        self.daily_trade_vals[date] = self.daily_trade_vals[date].append(self.tracker)
    #    except KeyError:
    #        self.daily_trade_vals[date] = self.tracker

    #def buy(self, date):
    #    #initializing tracker to keep up with trades
    #    self.tracker = []
    #    if len(self.daily_trade_vals) == 0:
    #        self.prev = self.port_val + self.value

    #    ranks = (self.alpha.loc[:,date].dropna().rank(pct = True) -0.5).sort_values(ascending = False, na_position = 'last')
    #    stock_top = ranks.iloc[0:self.top_num]
    #    stock_bot = ranks.dropna().iloc[-self.top_num:].abs()

    #    self.price_check_buy(symbols = stock_top, date = date) #check before calculating weight
    #    self.price_check_buy(symbols = stock_bot, date = date)
    #    self.tracker.append({"top_stocks":stock_top.append(stock_bot)})

    #    weighted_vals = []
    #    ref_check = M2D_F4.__read__(path = '/Users/alexq/Downloads/memmap/sample_weight.m2d_f4', sd = 20161230, ed = 20200102).loc[:,date]
    #    for row in ref_check.index:
    #        if ref_check.loc[row] != 0.0 and not np.isnan(ref_check.loc[row]):
    #            weighted_vals.append(row)

    #    weight = ref_check.loc[weighted_vals].abs()
    #    #calculating and recording the amount invested per stock
    #    target = weight*self.value
    #    #target = target_top
    #    self.tracker.append({"target":target})

    #    #determining how many shares of each stock, recording it in both our daily tracker and portfolio
    #    assert self.value >= 0
    #    position = target/self.open.loc[target.index, date]
    #    position = position.dropna(axis=0, how='any', inplace=False)
    #    temp = {}
    #    pos_dict = position.T.to_dict()

    #    sum = 0
    #    for symbol in self.portfolio: 
    #        if np.isnan(self.close.loc[symbol,date]):
    #            sum = sum + self.portfolio[symbol] * self.close.loc[symbol,:date].fillna(method='ffill').iloc[-1] #repetitive with the part in sell


    #    for key in self.portfolio:
    #        temp[key] = self.portfolio[key] + pos_dict.get(key, 0) 
    #    for key in pos_dict:
    #        if key not in temp:
    #            temp[key] = pos_dict[key]
    #    self.portfolio = temp
    #    #self.portfolio.update(position.T.to_dict()) #need to avoid mutating existing stock
    #    self.tracker.append({"position":position})
    #    #recording and finding out the total monetary value of my stocks
    #    close_val = position*self.close.loc[position.index,date] #sell value of all my stocks
    #    self.port_val = sum + self.port_val
    #    self.port_val = close_val.sum() + self.port_val
    #    self.tracker.append({"close":close_val})
    #     #using portfolio value - the net_value for the temporary delta and then changing the net_value
    #    self.value = 0
    #    delta_value = self.port_val - self.prev 
    #    self.net_val = self.net_val + delta_value
    #    self.cummulative_pnl.loc[date,'pnl'] = self.net_val #set the net_val into our P&L
    #    self.prev = self.port_val
    #    self.tracker.append({"delta_val":delta_value})
    #    self.tracker.append({"net_val":self.net_val})
    #    try:
    #        self.daily_trade_vals[date].append(self.tracker)
    #    except KeyError:
    #        self.daily_trade_vals[date] = self.tracker
        
        #pnl end of day
        
    def sell(self,date):
        temp = self.open.loc[self.portfolio.keys(),date]
        for row in self.open.loc[self.portfolio.keys(),date].index: #calculate isnan values here instead?
            temp.loc[row] = temp.loc[row]*self.portfolio.get(row)
        open_val = temp.loc[self.portfolio.keys()]
        self.port_val = open_val.sum()
        #price check and sell stocks
        self.price_check_sell(portfolio = self.portfolio, sell_price = open_val, date = date)
        try:
            self.daily_trade_vals[date] = self.daily_trade_vals[date].append(self.tracker)
        except KeyError:
            self.daily_trade_vals[date] = self.tracker
    

    def do_nothing(self, date): #record and modify all trade value data. update so it works with new trade function
        self.tracker = []
        self.prev = self.port_val + self.value
        reference = self.daily_trade_vals[list(self.daily_trade_vals)[-1]]

        #top stocks, weight, target, position, close_val, delta_val, net_val
        try:
            self.tracker.append({"top_stocks":reference[0].get("top_stocks")}) #or maintain previous days stocks
        except TypeError:
            self.tracker[0] = ({"top_stocks":reference[0].get("top_stocks")})

        updated_alpha = self.alpha.loc[reference[0].get("top_stocks").index, date]
        updated_alpha = updated_alpha.rank(ascending = False, na_option = 'keep', numeric_only = True, pct = True)
        updated_weight = updated_alpha.loc[:]/updated_alpha.loc[:].sum()
        self.tracker.append({"weight":updated_weight})

        self.tracker.append({"target":reference[2].get("target")})

        self.tracker.append({"position":reference[3].get("position")}) #stays constant

        #overwrite nans with last existing closing value somehow
        updated_close_sum = reference[3].get("position") * self.close.loc[reference[3].get("position").index, date]
        updated_delta = updated_close_sum.sum() - self.port_val
        self.port_val = updated_close_sum.sum()
        self.tracker.append({"close_val":updated_close_sum}) #this is still calculated based off previous weights
        self.tracker.append({"delta_val":updated_delta}) #delta should also be real values
        self.net_val = self.net_val+updated_delta
        self.prev = self.port_val
        self.tracker.append({"net_val":self.net_val+updated_delta}) #net_val should be mutated based off of the previous days weights
        #self.cummulative_pnl.loc[date,'pnl'] = self.net_val
        self.daily_trade_vals[date] = self.tracker

    def new_trade(self, date): 
        if len(self.daily_trade_vals) % self.freq == 0 or self.freq == 1:
            self.sell(date = date)
            self.buy(date = date)
        else: 
            self.do_nothing(date = date)
    
    #def ep_test(self, date, epsilon): #instead of using long short long is just our regular stock short is our optimal return
    #    for e in epsilon:
    #        stock_buy = self.alpha.loc[:,date].rank(ascending = False, na_option = 'keep', numeric_only = True, pct = True).sort_values(ascending = False, na_position = 'last').iloc[0:self.top_num]
    #        #need to find best performing stock
    #        temp = self.close.copy()
    #        temp.iloc[:,:] = temp.iloc[:,:] - self.open.iloc[:,:]
    #        temp.loc[:,date].sort_values(ascending = False)
    #        stock = temp.iloc[0,0]

    #        #perform calculations
    #        weight_pos = (stock_buy/stock_buy.sum()) * (1.0-e)
    #        weight_neg = (stock/stock) * e  # just e since only one stock to max profits

    #        target_pos = weight_pos*self.value_LS
    #        target_neg = weight_neg*self.value_LS

    #        position_pos = target_pos/self.open.loc[target_pos.index, date]
    #        position_neg = target_neg/self.open.loc[temp.loc[:,date].sort_values(ascending = False).index[0], date]
    #        close_val_pos = position_pos*self.close.loc[position_pos.index,date]
    #        close_val_neg = position_neg*self.close.loc[temp.loc[:,date].sort_values(ascending = False).index[0],date]
    #        delta_value_pos = close_val_pos.sum() - (self.value_LS*(1.0-e))
    #        delta_value_neg = close_val_neg.sum() - (self.value_LS*e)
    #        self.net_val_longshort = self.net_val_longshort + delta_value_pos + delta_value_neg
    #        self.cummulative_pnl_LS.loc[date,'pnl'] = self.net_val_longshort
    #        self.value_LS = close_val_pos.sum() + close_val_neg.sum()


    def backtest(self):
        self.buy(date = self.alpha.columns[0])
        for date in self.alpha.columns[1:]: #number of columns from start val to end val
            self.new_trade(date = date)
            #self.ep_test(date = date, epsilon = epsilon)

#if __name__ == '__main__':
#    #starts 20161230 ends 20200102
#    df_alphas = pd.read_csv("/Users/alexq/Downloads/20230713_153849_178287+20161230+20200101/railgun_tree_model_xgb_7_weight.csv") 
#    df_symbols = pd.read_csv("/Users/alexq/Downloads/20230713_153849_178287+20161230+20200101/railgun_tree_model_xgb_7_symbol.csv")
#    df_main = pd.read_pickle("/Users/alexq/Downloads/memmap/indicator.pkl").T

#    df_alphas.iloc[:,0] = pd.to_datetime(df_alphas.iloc[:,0])
#    df_alphas.iloc[:,0] = df_alphas.iloc[:,0].dt.strftime('%Y%m%d').astype(int)

#    df_symbols.iloc[:,0] = pd.to_datetime(df_symbols.iloc[:,0])
#    df_symbols.iloc[:,0] = df_symbols.iloc[:,0].dt.strftime('%Y%m%d').astype(int)

#    df_main.loc[:,:] = np.nan

#    for row in df_symbols.index:
#        print(row)
#        for col in df_symbols.columns[1:]:
#            df_main.loc[df_symbols.loc[row,col], df_alphas.iloc[int(row),0]] = df_alphas.loc[row,col]

#    with open(os.path.join('/Users/alexq/Downloads/memmap/', 'sample_weight.m2d_f4'), 'w') as fp:
#        pass

#    M2D_F4.__save__(df_main, path = "/Users/alexq/Downloads/memmap/sample_weight.m2d_f4")

#    a = 1

#current minimum sd value is 20170120 until i download more data.
                                      
if __name__ == '__main__':
    init_value = 1000
    #20170120
    test = backtester(sd = 20090123, ed = 20230531, lib_path = '/Users/alexq/Downloads/memmap/memmap2d/', top_num = 6, value = init_value, freq = 1)
    test.backtest()
    test.cummulative_pnl.reset_index()
    test.cummulative_pnl.index = pd.to_datetime(pd.Series(test.cummulative_pnl.index), format = "%Y%m%d")
    pnl = test.cummulative_pnl
    returns = pnl + init_value
    returns.columns  = ['return']
    returns = returns.pct_change()
    qs.reports.html(returns = returns['return'], output = 'stats.html', title = 'Strategy Tearsheet')

#def get_index_values(df:pd.DataFrame):
#    df = df.loc[:,"证券代码"] #list of symbols

#    path = '/Users/alexq/Downloads/memmap/memmap2d/index_val/'
#    file = ['index_open_df.m2d_f4','index_high_df.m2d_f4','index_low_df.m2d_f4','index_close_df.m2d_f4','index_volume_df.m2d_f4', 'index_amount_df.m2d_f4']
#    info = ['开盘指数', '最高指数', '最低指数', '收盘指数', '成交量', '成交额']

#    for i in range(len(file)):
#        with open(os.path.join(path, file[i]), 'w') as fp:
#            pass
    
#    processes = []
#    for row in df: #thread
#        symbols = row[0:6] 
#        process = multiprocessing.Process(target = ak.index_level_one_hist_sw, args = (symbols))
#        processes.append(process)
#    for p in processes:
#        p.start()
#    for p in processes:
#        result = p.join()
#        print(f'ak.index_level_one_hist_sw({p} = {result})'.join())

def get_index_values(df:pd.DataFrame):
    df = df.loc[:,"证券代码"] #list of symbols

    path = '/Users/alexq/Downloads/memmap/memmap2d/index_val/'
    file = ['index_open_df.m2d_f4','index_high_df.m2d_f4','index_low_df.m2d_f4','index_close_df.m2d_f4','index_volume_df.m2d_f4', 'index_amount_df.m2d_f4']
    info = ['开盘指数', '最高指数', '最低指数', '收盘指数', '成交量', '成交额']

    for i in range(len(file)):
        with open(os.path.join(path, file[i]), 'w') as fp:
            pass

    for k in range(len(file)):
        M2D_F4.__save__(pd.DataFrame(index = ["000001"], columns = [10000101]), path =path+file[k] )

    for row in df: #thread
        print(row)
        symbols = row[0:6] 
        table = ak.index_level_one_hist_sw(symbols)
        table['发布日期'] = pd.to_datetime(table.发布日期)
        table['发布日期'] = table['发布日期'].dt.strftime('%Y%m%d').astype(int)
        for j in range(len(file)):
            temp = table.pivot('指数代码','发布日期',info[j])
            M2D_F4.__upsert__(temp, path = path+file[j])

    for l in range(len(file)):
        df = M2D_F4.__read__(path = path + file[l])
        M2D_F4.__save__(df.iloc[1:,1:],path = path+file[l])

#if __name__ == '__main__':
#    df = pd.read_excel(io = '/Users/alexq/Downloads/memmap/memmap2d/givenfiles/sw_i_code.xlsx')
#    get_index_values(df)
    
    
    
