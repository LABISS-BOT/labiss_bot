import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from datetime import datetime
pd.core.common.is_list_like = pd.api.types.is_list_like
import schedule as sch
import datetime
import telegram

TEST_MODE = 3
# TEST_MODE
# 0 : logic_test
# 1 : logic_test + telegram test
# 2 : real test (매수 매도 진행)
# 3 : back test

if (TEST_MODE > 0):
    # telegram bot labiss_Bot
    bot = telegram.Bot(token='1889249363:AAH08lmqQcWh-CXJebrWAWNIYjwj-elwPZA')
    chat_id_user1 = 1629806638
    chat_id_user2 = 1818633385
    chat_id_user3 = 1847689887

# stationarity_test
# TODO 꼭 필요한가?
def stationarity_test(X, cutoff=0.01):
    # H_0 in adfuller is unit root exists (non-stationary)
    # We must observe significant p-value to convince ourselves that the series is stationary
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name +' is likely stationary.')
    else:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name +' is likely non-stationary.')

def show_plot(spread_df, title):
    print('show plot')
    # spread_df의 평균과 그래프
    ax = plt.gca()
    plt.title(title)
    plt.xlabel('5m candle')
    plt.ylabel(title)
    spread_df.plot(kind='line', x='datetime', y='spread', ax=ax)
    spread_df.plot(kind='line', x='datetime', y='mean', ax=ax, color='black', linestyle='--')
    spread_df.plot(kind='line', x='datetime', y='std_plus', ax=ax, color='green', linestyle='--')
    spread_df.plot(kind='line', x='datetime', y='std_minus', ax=ax, color='red', linestyle='--')
    # img 저장
    # file_path 최적화 필요
    filename = "".join(i for i in title if i not in "\/:*?<>|")
    file_path = 'spread_img\\'+filename+'.png'
    plt.savefig(file_path)
    if (TEST_MODE == 0 or TEST_MODE == 3):
        plt.show()
    plt.close()
    if (TEST_MODE == 1 or TEST_MODE == 2):
        bot.sendPhoto(chat_id_user1, photo=open(file_path,'rb'))
        bot.sendPhoto(chat_id_user2, photo=open(file_path,'rb'))
        bot.sendPhoto(chat_id_user3, photo=open(file_path, 'rb'))

def myrange(start, end, step):
    r = start
    while (r < end):
        yield r
        r += step

def find_coint_coefficient(pair_df, pair1, pair2):
    spread_df = pd.DataFrame()
    spread_df['datetime'] = pair_df['datetime']
    pvalue = 1
    min_cc = 0
    for i in myrange(0.1, 2.1, 0.1):
        spread_df['spread'] = np.log(pair_df[pair1]) - i * np.log(pair_df[pair2])
        spread_df2 = spread_df.iloc[:, 1:]
        if pvalue > adfuller(spread_df2)[1]:
            pvalue = adfuller(spread_df2)[1]
            min_cc = i
            stable_spread_df = spread_df.copy()
        else:
            pvalue = pvalue
    print("pvalue min = ", pvalue, "CC = ", min_cc)
    return stable_spread_df['spread'], min_cc

# compute log spread
def compute_log_spread(pair_df, pair1, pair2, score):
    # spread_df : spread 저장하는 pandas dataframe
    spread_df = pd.DataFrame()
    spread_df['datetime'] = pair_df['datetime']
    spread_df[pair1] = pair_df[pair1]
    spread_df[pair2] = pair_df[pair2]
    spread_df['spread'], CC = find_coint_coefficient(pair_df, pair1, pair2)
    # moving average window
    #TODO 뒤에서 몇개 가져올지
    window_ = 300
    spread_df['mean'] = spread_df['spread'].rolling(window_).mean()
    spread_df['std'] = spread_df['spread'].rolling(window_).std()
    spread_df['std_plus'] = spread_df['mean'] + spread_df['std']
    spread_df['std_minus'] = spread_df['mean'] - spread_df['std']
    title = str(pair_df['datetime'][0]) + '_' + str(pair1) + '-' + str(pair2) + ' SPREAD'
    show_plot(spread_df, title)

# find cointegrated_pairs for 25 coins in USDT market
def find_cointegrated_pairs(data):
    n = data.shape[1]
    print('n', n)
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            # 0.05 면 충분히 작은가?
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

# market data 긁어오기
#TODO 변동성이 작을 경우 너무 많은 페어를 잡음
def market_search(market_search_space):
    print("market search ...")
    pair_df = pd.DataFrame()
    for market_key in market_search_space:
        # TODO
        # USDT 포함된 마켓 키 골라서 거래량으로 sorting해서 pandas column에 추가 -> find_cointegrated_pair(pd)
        # 페어 찾고 그걸로 트레이딩
        # 1d ? 5m ??
        ohlcv = binance.fetch_ohlcv(symbol=market_key, timeframe='5m', limit=500)
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        pair_df['datetime'] = df['datetime']
        pair_df[market_key] = df['close']
        # print(pair_df)
        # pair_df.set_index(df['datetime'], inplace=True)
    print(pair_df.iloc[:, 1:26])
    # 500 개 보여줌 과거 데이터
    # TODO pairs 찾을 때 최적화 필요
    scores, pvalues, pairs = find_cointegrated_pairs(pair_df.iloc[:, 1:26])
    # print(scores)
    print(pairs)
    print(np.shape(scores))
    for pair in pairs:
        # pair to index
        # pair_index
        i = market_search_space.index(pair[0])
        j = market_search_space.index(pair[1])
        print(pair[0], i)
        print(pair[1], j)
        print(scores[i][j])
        compute_log_spread(pair_df, pair[0] ,pair[1], scores[i][j])
    print("Find pair : " ,len(pairs))

# leverage = 배율
# candle_count = 5분봉 몇개
# money
def back_test(money, leverage, candle_count, since=None):
    print('********* [LABISS Backtest] *********')
    print('현재 잔고 : ', money, 'USDT, leverage : x', leverage, ' Number of Candle : ', candle_count)
    pair_df = pd.DataFrame()
    pair1 = 'ETH/USDT'
    pair2 = 'AAVE/USDT'
    for market_key in [pair1, pair2]:
        ohlcv = binance.fetch_ohlcv(symbol=market_key, timeframe='5m', limit=candle_count)
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        pair_df['datetime'] = df['datetime']
        pair_df[market_key] = df['close']
    pair_df['spread'], CC = find_coint_coefficient(pair_df, pair1, pair2)

    pair_df['mean'] = pair_df['spread'].mean()
    pair_df['std'] = pair_df['spread'].std()
    pair_df['std_plus'] = pair_df['mean'] + pair_df['std']
    pair_df['std_minus'] = pair_df['mean'] - pair_df['std']
    title = str(pair_df['datetime'][0]) + '_' + str(pair1) + '-' + str(pair2) + ' SPREAD'

    mean = pair_df['spread'].mean()
    std = pair_df['spread'].std()
    std_plus = mean+std
    std_minus = mean-std
    eat_count = 0
    mean_index_array=[]
    std_index_array=[]
    for i in range(candle_count-1):
        a = pair_df['spread'][i]
        b = pair_df['spread'][i+1]
        if ((mean-a)*(mean-b)<=0):
            mean_index_array.append(i+1)
        if (((std_plus-a)*(std_plus-b)<=0) or ((std_minus-a)*(std_minus-b)<=0)):
            std_index_array.append(i)
    print('length of mean_index_array : ', len(mean_index_array))
    print('mean_index_array', mean_index_array)
    print('length of std_index_array : ', len(std_index_array))
    print('std_index_array', std_index_array)

    for i, mean_ in enumerate(mean_index_array[:-1]):
        for std_ in std_index_array:
            print('test std_ : ',std_, ' mean[i] : ',mean_index_array[i], ' mean[i+1] : ', mean_index_array[i+1])
            if (std_ >= mean_index_array[i] and std_ <= mean_index_array[i + 1]):
                eat_count += 1
                print('eat_count : ', eat_count)
                break

    print('eat_count : ', eat_count)

    A1 = pair_df[pair1][mean_index_array[0]]
    B1 = pair_df[pair2][mean_index_array[0]]
    A2 = pair_df[pair1][std_index_array[0]]
    B2 = pair_df[pair2][std_index_array[0]]
    print(A1, B1, A2, B2)
    money1 = money2 = money/2
    print(money1, money2)
    profit = (A1/A2 + B2/B1)*eat_count

    print('before : ', money, ' after : ', money+profit , ' ratio : ', (money+profit)/(money)*100, '%')


    # 수익률
    # pair_df['ror'] = np.where(pair_df['spread'] > pair
    #
    # )
    show_plot(pair_df, title)
    print(pair_df)

if __name__ == '__main__':
    binance = ccxt.binance()
    #TODO 거래량 상위 30개 긁어오기
    # markets = binance.load_markets()
    # print(markets.keys())
    # print(len(markets))
    market_search_space = ['ETH/USDT','BTC/USDT','BNB/USDT','XRP/USDT','MATIC/USDT','ADA/USDT','DOT/USDT',
        'WRX/USDT','LINK/USDT','VET/USDT','EOS/USDT','LTC/USDT','ETC/USDT','UNI/USDT','SOL/USDT','CAKE/USDT','THETA/USDT',
        'AAVE/USDT','FIL/USDT','BCH/USDT','LUNA/USDT','TRX/USDT','XLM/USDT','SXP/USDT','BAKE/USDT']
    if (TEST_MODE != 3):
        market_search(market_search_space)
    else:
        back_test(100, 5, 100)
    # sch.every(5).minutes.do(market_search,market_search_space)

    # while True:
    #     sch.run_pending()
    #     time.sleep(??)

    # (일자, 시가, 고가, 저가, 종가, 거래량)
    # print("date 시가 고가 저가 종가 거래량")
    # for ohlc in ohlcvs:
    #     print(datetime.fromtimestamp(ohlc[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'), ohlc[1], ohlc[2], ohlc[3],ohlc[4], ohlc[5])

    # orderbook = binance.fetch_order_book('BTC/USDT')
    # bids 매수 asks 매도 100개 보여줌
    # orderbook (가격, 수량)
    # print(orderbook['symbol'])
    # print(len(orderbook['bids']))
    # print(orderbook['bids'][0][1])
    # print(orderbook['bids'][99][1])
    # for ask in orderbook['asks']:
    #     print(ask[0], ask[1])