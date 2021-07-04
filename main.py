import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from datetime import datetime
pd.core.common.is_list_like = pd.api.types.is_list_like
import schedule as sch
import telegram
import calendar

TEST_MODE = 3
# TEST_MODE
# 0 : logic_test
# 1 : logic_test + telegram test
# 2 : real test (매수 매도 진행)
# 3 : back test
SHOW_PLOT = 0

money = 100
changes = 100
# key : (pair1, pair2)
# value : (buy_long_symbol, buy_short_symbol, buy_long_price, buy_short_price, buy_long_num, buy_short_num)
pair_dict = {}

if (TEST_MODE > 0):
    # telegram bot labiss_Bot
    bot = telegram.Bot(token='1889249363:AAH08lmqQcWh-CXJebrWAWNIYjwj-elwPZA')
    chat_id_user1 = 1629806638
    chat_id_user2 = 1818633385
    chat_id_user3 = 1847689887

# graph 그리기
def show_plot(spread_df, mean, std_plus, std_minus, title):
    if SHOW_PLOT:
        # print('show plot')
        # spread_df의 평균과 그래프
        ax = plt.gca()
        plt.title(title)
        plt.xlabel('5m candle')
        plt.ylabel(title)
        spread_df.plot(kind='line', x='datetime', y='spread', ax=ax)
        std = std_plus - mean
        plt.axhline(y=mean, color='black', linestyle='--')
        plt.axhline(y=std_plus, color='green', linestyle='--')
        plt.axhline(y=std_minus, color='red', linestyle='--')
        plt.axhline(y=std_plus + std, color='green')
        plt.axhline(y=std_minus - std, color='red')
        # img 저장
        # file_path 최적화 필요
        filename = "".join(i for i in title if i not in "\/:*?<>|")
        file_path = 'spread_img\\'+filename+'.png'
        plt.savefig(file_path)
        if (TEST_MODE == 0 or TEST_MODE == 3):
            plt.show()
            # plt.draw()
            # plt.waitforbuttonpress(0)
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
    # print(pair_df)
    for i in myrange(0.1, 2.1, 0.1):
        spread_df['spread'] = np.log(pair_df[pair1]) - i * np.log(pair_df[pair2])
        spread_df2 = spread_df.iloc[:, 1:].values
        # print(spread_df2)
        if pvalue > adfuller(spread_df2, autolag='AIC')[1]:
            pvalue = adfuller(spread_df2, autolag='AIC')[1]
            min_cc = i
            stable_spread_df = spread_df.copy()
        else:
            pvalue = pvalue
    # print("pvalue min = ", pvalue, "CC = ", min_cc)
    return stable_spread_df['spread'], min_cc, pvalue

# compute log spread
def compute_log_spread(pair_df, pair1, pair2, score):
    # spread_df : spread 저장하는 pandas dataframe
    spread_df = pd.DataFrame()
    spread_df['datetime'] = pair_df['datetime']
    spread_df[pair1] = pair_df[pair1]
    spread_df[pair2] = pair_df[pair2]
    spread_df['spread'], CC, pvalue = find_coint_coefficient(pair_df, pair1, pair2)
    # moving average window
    #TODO 뒤에서 몇개 가져올지
    mean = spread_df['spread'].mean()
    std = spread_df['spread'].std()
    std_plus = mean+std
    std_minus = mean-std
    title = str(pair_df['datetime'][0]) + '_' + str(pair1) + '-' + str(pair2) + ' SPREAD'
    show_plot(pair_df, mean, std_plus, std_minus, title)

# find cointegrated_pairs for 25 coins in USDT market
def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    # pvalue_matrix = np.ones((n, n))
    pvalue_dict = {}
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
            # pvalue_matrix[i, j] = pvalue
            # 0.05 면 충분히 작은가?
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
                pvalue_dict[(keys[i], keys[j])] = pvalue
    return score_matrix, pvalue_dict, pairs

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

# def select_action(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue):
#     # action
#     # 0 == 아무것도, 1 == 매수, 2 == 수익실현, 3 == 손절,
#     curr_value = pair_df['spread'].iloc[-1]
#     past_value = pair_df['spread'].iloc[-2]
#     #print(curr_value)
#     print('==================== select_action ====================')
#     # global position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money, pair_dict
#     global pair_dict, money
#     # 수익실현
#     # TODO 현재 position은 0,1 페어가 추가되면 정수가 될것
#     position = len(pair_dict)
#     if (pair1, pair2) in pair_dict:
#         buy_long_symbol = pair_dict[(pair1, pair2)][0]
#         buy_short_symbol = pair_dict[(pair1, pair2)][1]
#         buy_long_price = pair_dict[(pair1, pair2)][2]
#         buy_short_price = pair_dict[(pair1, pair2)][3]
#         buy_long_num = pair_dict[(pair1, pair2)][4]
#         buy_short_num = pair_dict[(pair1, pair2)][5]
#         print('position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money')
#         print(position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money)
#     else:
#         print('no position money : ', money)
#
#     if position == 1 and ((mean - past_value) * (mean - curr_value) <= 0):
#         print('+++++++++++++good sell +++++++++++++')
#         money += (buy_long_price - pair_df[buy_long_symbol].iloc[-1]) * buy_long_num \
#                  - (buy_short_price - pair_df[buy_short_symbol].iloc[-1]) * buy_short_num
#         # position = 0
#         del pair_dict[(pair1, pair2)]
#     # 손절 (일단 2 sigma) Warning Value
#     if position == 1 and ((pvalue > 0.05) or (curr_value >= mean+2*std or curr_value <= mean-2*std)):
#         print('+++++++++++++10 sell+++++++++++++')
#         money += (buy_long_price-pair_df[buy_long_symbol].iloc[-1])*buy_long_num \
#                 - (buy_short_price-pair_df[buy_short_symbol].iloc[-1])*buy_short_num
#         # position = 0
#         del pair_dict[(pair1, pair2)]
#     # 매수
#     if position == 0 and (curr_value >= std_plus) and ((pvalue < 0.05) and (curr_value <= mean+2*std and curr_value >= mean-2*std)):
#         print('+++++++++++++buy+++++++++++++')
#         buy_long_symbol = pair_df.columns.values[2]
#         buy_short_symbol = pair_df.columns.values[1]
#         buy_long_price = pair_df[buy_long_symbol].iloc[-1]
#         buy_short_price = pair_df[buy_short_symbol].iloc[-1]
#         buy_long_num = money / (CC+1) / buy_long_price
#         buy_short_num = (money * CC) / (CC + 1) / buy_short_price
#         # position = 1
#         # (buy_long_symbol, buy_short_symbol, buy_long_price, buy_short_price, buy_long_num, buy_short_num)
#         pair_dict[(pair1, pair2)] = (buy_long_symbol,buy_short_symbol, buy_long_price, buy_short_price, buy_long_num, buy_short_num)
#     elif position == 0 and (curr_value <= std_minus) and ((pvalue < 0.05) and (curr_value <= mean+2*std and curr_value >= mean-2*std)):
#         print('+++++++++++++buy+++++++++++++')
#         buy_long_symbol = pair_df.columns.values[1]
#         buy_short_symbol = pair_df.columns.values[2]
#         buy_long_price = pair_df[buy_long_symbol].iloc[-1]
#         buy_short_price = pair_df[buy_short_symbol].iloc[-1]
#         buy_long_num = (money * CC) / (CC + 1) / buy_long_price
#         buy_short_num = money / (CC + 1) / buy_short_price
#         # position = 1
#         pair_dict[(pair1, pair2)] = (buy_long_symbol,buy_short_symbol, buy_long_price, buy_short_price, buy_long_num, buy_short_num)

def select_action(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue, leverage):
    # action
    # 0 == 아무것도, 1 == 매수, 2 == 수익실현, 3 == 손절,
    curr_value = pair_df['spread'].iloc[-1]
    past_value = pair_df['spread'].iloc[-2]
    #print(curr_value)
    # print('==================== select_action ====================')
    # global position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money, pair_dict
    global pair_dict, money, changes
    # 수익실현
    position = len(pair_dict)
    this_money = changes / (11 - position)
    print('changes : ', changes, ' position : ' , position)
    this_position_1 = (pair1, pair2) in pair_dict
    this_position_2 = (pair2, pair1) in pair_dict
    this_position = this_position_1 or this_position_2
    # TODO pair1, pair2 가 바뀌엇을때 어쩌지?? 중요!!
    # print('pair1 : ', pair1, ' pair2 : ', pair2, ' this_position :', this_position)
    # if (pair1, pair2) in pair_dict:
    if this_position_1:
        buy_long_symbol = pair_dict[(pair1, pair2)][0]
        buy_short_symbol = pair_dict[(pair1, pair2)][1]
        buy_long_price = pair_dict[(pair1, pair2)][2]
        buy_short_price = pair_dict[(pair1, pair2)][3]
        buy_long_num = pair_dict[(pair1, pair2)][4]
        buy_short_num = pair_dict[(pair1, pair2)][5]
    elif this_position_2:
        buy_long_symbol = pair_dict[(pair2, pair1)][0]
        buy_short_symbol = pair_dict[(pair2, pair1)][1]
        buy_long_price = pair_dict[(pair2, pair1)][2]
        buy_short_price = pair_dict[(pair2, pair1)][3]
        buy_long_num = pair_dict[(pair2, pair1)][4]
        buy_short_num = pair_dict[(pair2, pair1)][5]

        # print('position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money')
        # print(position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money)
    # else:
        # print('no position money : ', money)

    if this_position == 1 and ((mean - past_value) * (mean - curr_value) <= 0):
        print('+++++++++++++good sell +++++++++++++')
        money += (buy_long_price - pair_df[buy_long_symbol].iloc[-1]) * buy_long_num \
                 - (buy_short_price - pair_df[buy_short_symbol].iloc[-1]) * buy_short_num
        changes += (pair_df[buy_long_symbol].iloc[-1]) * buy_long_num \
                 + (pair_df[buy_short_symbol].iloc[-1]) * buy_short_num
        # position = 0
        print('Show me the money : ', money)

        if this_position_1:
            del pair_dict[(pair1, pair2)]
        elif this_position_2:
            del pair_dict[(pair2, pair1)]

    # 손절 (일단 2 sigma) Warning Value
    if this_position == 1 and ((pvalue > 0.05) or (curr_value >= mean+2*std or curr_value <= mean-2*std)):
        print('+++++++++++++10 sell+++++++++++++')
        money += (buy_long_price-pair_df[buy_long_symbol].iloc[-1])*buy_long_num \
                - (buy_short_price-pair_df[buy_short_symbol].iloc[-1])*buy_short_num
        changes += (pair_df[buy_long_symbol].iloc[-1]) * buy_long_num \
                 + (pair_df[buy_short_symbol].iloc[-1]) * buy_short_num
        print('Show me the money : ', money)
        # position = 0
        if this_position_1:
            del pair_dict[(pair1, pair2)]
        elif this_position_2:
            del pair_dict[(pair2, pair1)]
    # 매수
    if this_position == 0 and (curr_value >= std_plus) and ((pvalue < 0.05) and (curr_value <= mean+2*std and curr_value >= mean-2*std)):
        print('+++++++++++++buy+++++++++++++')
        # buy_long_symbol = pair_df.columns.values[2]
        # buy_short_symbol = pair_df.columns.values[1]
        buy_long_symbol = pair2
        buy_short_symbol = pair1
        buy_long_price = pair_df[buy_long_symbol].iloc[-1]
        buy_short_price = pair_df[buy_short_symbol].iloc[-1]
        buy_long_num = this_money / (CC+1) / buy_long_price * leverage
        buy_short_num = (this_money * CC) / (CC + 1) / buy_short_price * leverage
        changes -= buy_long_price * buy_long_num + buy_short_price * buy_short_num
        # position = 1
        # (buy_long_symbol, buy_short_symbol, buy_long_price, buy_short_price, buy_long_num, buy_short_num)
        print('position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money')
        print(position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num,money)
        pair_dict[(pair1, pair2)] = (buy_long_symbol,buy_short_symbol, buy_long_price, buy_short_price, buy_long_num, buy_short_num)

    elif this_position == 0 and (curr_value <= std_minus) and ((pvalue < 0.05) and (curr_value <= mean+2*std and curr_value >= mean-2*std)):
        print('+++++++++++++buy+++++++++++++')
        # buy_long_symbol = pair_df.columns.values[1]
        # buy_short_symbol = pair_df.columns.values[2]
        buy_long_symbol = pair1
        buy_short_symbol = pair2
        buy_long_price = pair_df[buy_long_symbol].iloc[-1]
        buy_short_price = pair_df[buy_short_symbol].iloc[-1]
        buy_long_num = (this_money * CC) / (CC + 1) / buy_long_price * leverage
        buy_short_num = this_money / (CC + 1) / buy_short_price * leverage
        changes -= buy_long_price * buy_long_num + buy_short_price * buy_short_num
        # position = 1
        print('position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money')
        print(position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num,money)
        pair_dict[(pair1, pair2)] = (buy_long_symbol,buy_short_symbol, buy_long_price, buy_short_price, buy_long_num, buy_short_num)

def select_buy(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue, leverage):
    # action
    # 0 == 아무것도, 1 == 매수, 2 == 수익실현, 3 == 손절,
    curr_value = pair_df['spread'].iloc[-1]
    past_value = pair_df['spread'].iloc[-2]
    # print(curr_value)
    # print('==================== select_action ====================')
    # global position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money, pair_dict
    global pair_dict, money, changes
    # 수익실현
    position = len(pair_dict)
    # print('changes : ', changes, ' position : ', position)
    if position > 9:
        return
    this_money = changes / (11 - position)

    # 매수
    if (curr_value >= std_plus) and ((pvalue < 0.05) and (curr_value <= mean + 2 * std and curr_value >= mean - 2 * std)):
        print('+++++++++++++buy+++++++++++++')
        # buy_long_symbol = pair_df.columns.values[2]
        # buy_short_symbol = pair_df.columns.values[1]
        buy_long_symbol = pair2
        buy_short_symbol = pair1
        buy_long_price = pair_df[buy_long_symbol].iloc[-1]
        buy_short_price = pair_df[buy_short_symbol].iloc[-1]
        buy_long_num = this_money / (CC + 1) / buy_long_price * leverage
        buy_short_num = (this_money * CC) / (CC + 1) / buy_short_price * leverage
        changes -= buy_long_price * buy_long_num + buy_short_price * buy_short_num
        # position = 1
        # (buy_long_symbol, buy_short_symbol, buy_long_price, buy_short_price, buy_long_num, buy_short_num)
        print('position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money')
        print(position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money)
        pair_dict[(pair1, pair2)] = (buy_long_symbol, buy_short_symbol, buy_long_price, buy_short_price, buy_long_num, buy_short_num)
        title = str(pair1) + "-" + str(pair2) + "-BUY"
        show_plot(pair_df, mean, std_plus, std_minus, title)

    elif (curr_value <= std_minus) and ((pvalue < 0.05) and (curr_value <= mean + 2 * std and curr_value >= mean - 2 * std)):
        print('+++++++++++++buy+++++++++++++')
        # buy_long_symbol = pair_df.columns.values[1]
        # buy_short_symbol = pair_df.columns.values[2]
        buy_long_symbol = pair1
        buy_short_symbol = pair2
        buy_long_price = pair_df[buy_long_symbol].iloc[-1]
        buy_short_price = pair_df[buy_short_symbol].iloc[-1]
        buy_long_num = (this_money * CC) / (CC + 1) / buy_long_price * leverage
        buy_short_num = this_money / (CC + 1) / buy_short_price * leverage
        changes -= buy_long_price * buy_long_num + buy_short_price * buy_short_num
        # position = 1
        print('position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money')
        print(position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num,money)
        pair_dict[(pair1, pair2)] = (buy_long_symbol, buy_short_symbol, buy_long_price, buy_short_price, buy_long_num, buy_short_num)
        title = str(pair1) + "-" + str(pair2) + "-BUY"
        show_plot(pair_df, mean, std_plus, std_minus, title)


def select_sell(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue, leverage):
    # position 항상 1
    # action
    # 0 == 아무것도, 1 == 매수, 2 == 수익실현, 3 == 손절,
    curr_value = pair_df['spread'].iloc[-1]
    past_value = pair_df['spread'].iloc[-2]
    # print(curr_value)
    # print('==================== select_action ====================')
    # global position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money, pair_dict
    global pair_dict, money, changes
    # 수익실현

    buy_long_symbol = pair_dict[(pair1, pair2)][0]
    buy_short_symbol = pair_dict[(pair1, pair2)][1]
    buy_long_price = pair_dict[(pair1, pair2)][2]
    buy_short_price = pair_dict[(pair1, pair2)][3]
    buy_long_num = pair_dict[(pair1, pair2)][4]
    buy_short_num = pair_dict[(pair1, pair2)][5]
    if (mean - past_value) * (mean - curr_value) <= 0:
        print('+++++++++++++good sell +++++++++++++')
        # money += (buy_long_price - pair_df[buy_long_symbol].iloc[-1]) * buy_long_num \
        #          - (buy_short_price - pair_df[buy_short_symbol].iloc[-1]) * buy_short_num
        # changes += (pair_df[buy_long_symbol].iloc[-1]) * buy_long_num \
        #            + (pair_df[buy_short_symbol].iloc[-1]) * buy_short_num
        money += -(buy_long_price - pair_df[buy_long_symbol].iloc[-1]) * buy_long_num \
                 + (buy_short_price - pair_df[buy_short_symbol].iloc[-1]) * buy_short_num
        changes += (pair_df[buy_long_symbol].iloc[-1]) * buy_long_num \
                   + (pair_df[buy_short_symbol].iloc[-1]) * buy_short_num

        # position = 0
        print('long :', buy_long_symbol, ' short :' , buy_short_symbol)
        print('long position : ', (buy_long_price - pair_df[buy_long_symbol].iloc[-1]) * buy_long_num, 'short position : ', (buy_short_price - pair_df[buy_short_symbol].iloc[-1]) * buy_short_num)
        print('Show me the money : ', money , ' change : ', changes)
        del pair_dict[(pair1, pair2)]
        title = str(pair1) + "-" + str(pair2) + "-SELL"
        show_plot(pair_df, mean, std_plus, std_minus, title)

    # 손절 (일단 2 sigma) Warning Value
    elif (pvalue > 0.05) or (curr_value >= mean + 2 * std or curr_value <= mean - 2 * std):
        print('+++++++++++++10 sell+++++++++++++')
        # money += (buy_long_price - pair_df[buy_long_symbol].iloc[-1]) * buy_long_num \
        #          - (buy_short_price - pair_df[buy_short_symbol].iloc[-1]) * buy_short_num
        # changes += (pair_df[buy_long_symbol].iloc[-1]) * buy_long_num \
        #            + (pair_df[buy_short_symbol].iloc[-1]) * buy_short_num
        money += -(buy_long_price - pair_df[buy_long_symbol].iloc[-1]) * buy_long_num \
                 + (buy_short_price - pair_df[buy_short_symbol].iloc[-1]) * buy_short_num
        changes += (pair_df[buy_long_symbol].iloc[-1]) * buy_long_num \
                   + (pair_df[buy_short_symbol].iloc[-1]) * buy_short_num
        print('long :', buy_long_symbol, ' short :', buy_short_symbol)
        print('long position : ', (buy_long_price - pair_df[buy_long_symbol].iloc[-1]) * buy_long_num,
              'short position : ', (buy_short_price - pair_df[buy_short_symbol].iloc[-1]) * buy_short_num)
        print('Show me the money : ', money, ' change : ', changes)
        # position = 0
        del pair_dict[(pair1, pair2)]
        title = str(pair1) + "-" + str(pair2) + "-SELL"
        show_plot(pair_df, mean, std_plus, std_minus, title)

def get_backdata_candle(since, candle_count, market_search_space):
    print(since)
    print("get backdata candle...")
    pair_df_raw_ = pd.DataFrame()
    start_dt = []
    since_dt = []
    for i in since:
        start_dt_ = datetime.strptime(i, "%Y%m%d")
        print('start_dt_:',start_dt_)
        start_dt.append(calendar.timegm(start_dt_.utctimetuple()) * 1000)
    # 5분봉 x 1000개 max candle * to ms
    interval = 5 * candle_count * 60000
    print('start_dt', start_dt, 'interval : ', interval, ' next : ', start_dt[0]+interval)
    # back data 가져오기
    for since in range(start_dt[0], start_dt[1], interval):
        # print('since : ',since)
        if since + interval > start_dt[1]:
            print('last since_interval : ', since +interval, ' start_dt[1] : ', start_dt[1])
            candle_count = candle_count - int((start_dt[1] - since) / (60000 * 5))
        pair_temp = pd.DataFrame()
        for market_key in market_search_space:
            ohlcv = binance.fetch_ohlcv(symbol=market_key, timeframe='5m', limit=candle_count, since=since)
            df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            # print('df : ',df)
            pair_temp['datetime'] = df['datetime']
            pair_temp[market_key] = df['close']
        pair_df_raw_ = pd.concat([pair_df_raw_, pair_temp], ignore_index=True)
        print(pair_df_raw_)
    return pair_df_raw_

def get_vaild_keys(sorted_pairs, pair_dict_keys):
    temp = []
    for i in range(0, len(sorted_pairs) - 1, 1):
        temp.append(sorted_pairs[i][0])
    # print("temp : " ,temp)
    # print("pair_dict_keys : ", pair_dict_keys)
    valid_keys = []
    for i in temp:
        if ((i[0],i[1]) not in pair_dict_keys) and ((i[1],i[0]) not in pair_dict_keys):
            valid_keys.append(i)
    # print("valid_keys : ",valid_keys)
    return valid_keys

# leverage = 배율
# candle_count = 5분봉 몇개
# money
def back_test(market_search_space, money, leverage, candle_count, window, since=None):
    print('********* [LABISS Backtest] *********')
    print('현재 잔고 : ', money, 'USDT, leverage : x', leverage, ' Number of Candle : ', candle_count, ' window : ', window)
    pair_df = pd.DataFrame()
    pair_df_raw = pd.DataFrame()

    # backdata 가져오기
    pair_df_raw_ = get_backdata_candle(since, candle_count, market_search_space)
    print('pair_df_raw_',pair_df_raw_)
    # pair 찾기

    init_money = money
    # window 이동시키면서 누적으로도 테스트 해봐야함
    candle_count = len(pair_df_raw_.index)
    for i in range(0, candle_count-window):
        print('step : ', i, '/', candle_count-window)
        # 현재 포지션에 대해서 select action
        for (pair1, pair2) in list(pair_dict.keys()):
            # print("pair1 : ", pair1, " pair2: ", pair2)
            pair_df_raw['datetime'] = pair_df_raw_['datetime']
            pair_df_raw[pair1] = pair_df_raw_[pair1]
            pair_df_raw[pair2] = pair_df_raw_[pair2]
            pair_df = pair_df_raw.iloc[i:i + window].copy()
            pair_df['spread'], CC, pvalue = find_coint_coefficient(pair_df, pair1, pair2)
            mean = pair_df['spread'].mean()
            std = pair_df['spread'].std()
            std_plus = mean + std
            std_minus = mean - std
            title = "test"
            # print(pair_df)
            # select_action(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue, leverage)
            # 팔지 결정
            select_sell(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue, leverage)
            # show_plot(pair_df, mean, std_plus, std_minus, title)
            # print('Show me the money : ', money)

        pair_number = len(pair_dict)
        print("pair_number", pair_number)
        print("pair_dict : ",pair_dict)
        if pair_number < 10:
            # 먼저 pair_dict key에 있는 페어로 select action
            # 끝나고 새로운 페어 찾기
            # pair_df, pair1, pair2 = find_pair()
            pair_df_ = pair_df_raw_.iloc[i:i+window].copy()
            # scores, pvalues, pairs = find_cointegrated_pairs(pair_df_.iloc[:, 1:26])
            scores, pvalues, pairs = find_cointegrated_pairs(pair_df_.iloc[:, 1:14])
            sorted_pairs = sorted(pvalues.items(), key=operator.itemgetter(1))
            valid_keys = get_vaild_keys(sorted_pairs, list(pair_dict.keys()))
            for (pair1, pair2) in valid_keys:
                # print("pair1 : ", pair1, " pair2: ", pair2)
                pair_df_raw['datetime'] = pair_df_raw_['datetime']
                pair_df_raw[pair1] = pair_df_raw_[pair1]
                pair_df_raw[pair2] = pair_df_raw_[pair2]
                pair_df = pair_df_raw.iloc[i:i + window].copy()
                pair_df['spread'], CC, pvalue = find_coint_coefficient(pair_df, pair1, pair2)
                mean = pair_df['spread'].mean()
                std = pair_df['spread'].std()
                std_plus = mean + std
                std_minus = mean - std
                title = "test"
                # print(pair_df)
                # select_action(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue, leverage)
                select_buy(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue, leverage)
                # show_plot(pair_df, mean, std_plus, std_minus, title)
        # print('Show me the money : ', money)

    print('Ratio', money/init_money*100)

if __name__ == '__main__':
    binance = ccxt.binance()
    #TODO 거래량 상위 30개 긁어오기
    # markets = binance.load_markets()
    # print(markets.keys())
    # print(len(markets))
    # market_search_space = ['ETH/USDT','BTC/USDT','BNB/USDT','XRP/USDT','MATIC/USDT','ADA/USDT','DOT/USDT',
    #     'WRX/USDT','LINK/USDT','VET/USDT','EOS/USDT','LTC/USDT','ETC/USDT','UNI/USDT','SOL/USDT','CAKE/USDT','THETA/USDT',
    #     'AAVE/USDT','FIL/USDT','BCH/USDT','LUNA/USDT','TRX/USDT','XLM/USDT','SXP/USDT','BAKE/USDT']
    market_search_space = ['ETH/USDT','BTC/USDT','XRP/USDT','ADA/USDT','LINK/USDT','VET/USDT','EOS/USDT','LTC/USDT','ETC/USDT',
        'BCH/USDT','TRX/USDT','XLM/USDT']
    if (TEST_MODE != 3):
        market_search(market_search_space)
    else:
        # money, leverage, candle_count, window
        date = ['20210530', '20210601']
        back_test(market_search_space, money, 1, 1000, 300, date)
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