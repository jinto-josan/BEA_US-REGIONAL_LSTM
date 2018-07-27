import datetime
import quandl
import numpy as np
class Get_data():

    def __init__(self):
        '''
        Class to get the quandl data to get annual us regional gdp growth.
        '''
        pass

    def gea(self, state):
        '''
        :indicators:is a string which can be GDP, PER CAPITA GDP, REAL GDP and so on
        :industry: can be Agriculture, Mining and so on, for all industry A
        :state: The state in US. Full name of the state in capitals, and space between should be removed
            example: south dakota should be written as SOUTHDAKOTA
        :return: A dataframe with data from BEA
        '''
        state=state.upper()
        state=state.replace(" ","")
        quandl.ApiConfig.api_key = "PuRfENCfAiNZn9_6Y4xL"
        dat1 = quandl.get('BEA/GSP_NAICS_ALL_C_'+state,start_date='1999-12-31', end_date='2014-12-31') #gdp all industries
        dat2 = quandl.get('BEA/GSP_NAICS_ALL_PC_'+state,start_date='1999-12-31', end_date='2014-12-31') #per capita gdp
        dat2.reset_index(inplace=True)
        dat2['Date'] = dat2['Date'].map(lambda x: x.year)
        dat2 = dat2.set_index('Date')
        dat1.reset_index(inplace=True)
        dat1['Date'] = dat1['Date'].map(lambda x: x.year)
        dat1 = dat1.set_index('Date')
        dat1.replace(np.nan, 0, inplace=True)
        dat2.replace(np.nan, 0, inplace=True)
        return dat1,dat2
