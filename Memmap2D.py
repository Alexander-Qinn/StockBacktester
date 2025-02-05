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

class backtester:


    def __init__(self, sd, ed, value, lib_path:str, top_num:int, freq:int):
        self.value = value
        self.tracker = []

        self.alpha = M2D_F4.__read__(path = lib_path + 'calculated_idx_val/idxalpha_val.m2d_f4', sd=sd, ed=ed) #index

        self.top_num = top_num
        self.net_val = 0

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
        
        self.tracker.append({"position":position})
        
        close_val = position*self.close.loc[position.index,date] #sell value of all my stocks
        self.port_val = sum + self.port_val
        self.port_val = close_val.sum() + self.port_val
        self.tracker.append({"close":close_val})
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

    def backtest(self):
        self.buy(date = self.alpha.columns[0])
        for date in self.alpha.columns[1:]: #number of columns from start val to end val
            self.new_trade(date = date)
            #self.ep_test(date = date, epsilon = epsilon)

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

    
    
    
