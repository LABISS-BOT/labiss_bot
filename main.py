import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
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
import gc
import tracemalloc as tc
import re
import time
# from ccxt.binance.exceptions import BinanceAPIException
tc.start(10)

TEST_MODE = 1
# TEST_MODE
# 0 : logic_test
# 1 : real time + back test
# 2 : real time + (매수 매도 진행)
# 3 : back test
SHOW_PLOT = 1

# 변동성 돌파 적용 = 1, 적용 x = 0
TARGET_MODE = 0

# moving average
MVG = 1

money = 100
changes = 100
cnt_trade = 0
cnt_good_sell = 0
cnt_10_sell = 0
# key : (pair1, pair2)
# value : (buy_long_symbol, buy_short_symbol, buy_long_price, buy_short_price, buy_long_num, buy_short_num, long_leverage, short_leverage, CC)
pair_dict = {}
# 매수 포지션 wait dict
# buy_wait_dict[buy_long_symbol] = (long_order['info']['orderId'], buy_long_num, buy_long_price, buy_long_num * buy_long_price)
buy_wait_dict = {}
# 매도 포지션 wait dict
# sell_wait_dict[buy_long_symbol] = (short_order['info']['orderId'], buy_short_num, buy_short_price, buy_short_num * buy_short_price)
sell_wait_dict = {}

if (TEST_MODE  >= 0):
    # telegram bot labiss_Bot
    # https://api.telegram.org/bot1889249363:AAH08lmqQcWh-CXJebrWAWNIYjwj-elwPZA/getUpdates
    bot = telegram.Bot(token='1889249363:AAH08lmqQcWh-CXJebrWAWNIYjwj-elwPZA')
    chat_id_user1 = 1629806638
    chat_id_user2 = 1818633385
    chat_id_user3 = 1847689887
    chat_id_user = [chat_id_user1, chat_id_user2, chat_id_user3]

def send_message(chat_id_user, txt):
    for id_ in chat_id_user:
        try:
            bot.sendMessage(id_,txt)
        except Exception as e:
            print(e)
            return

# graph 그리기
def show_plot(spread_df, mean, std_plus, std_minus, title, sigma):
    # time1 = tc.take_snapshot()
    if SHOW_PLOT:
        # print('show plot')
        # spread_df의 평균과 그래프
        ax = plt.gca()
        plt.title(title)
        plt.xlabel('5m candle')
        plt.ylabel(title)
        spread_df.plot(kind='line', x='datetime', y='zscore_60_5', ax=ax)
        std = std_plus - mean
        plt.axhline(0, color='black')
        plt.axhline(1.0, color='red', linestyle='--')
        plt.axhline(-1.0, color='green', linestyle='--')
        plt.axhline(3.5, color='black', linestyle='--')
        plt.axhline(-3.5, color='black', linestyle='--')
        # img 저장
        file_path = 'spread_img\\'+title+'.png'
        plt.savefig(file_path)
        # if (TEST_MODE == 0 or TEST_MODE == 3):
            # plt.show()
            # plt.draw()
            # plt.waitforbuttonpress(0)
        # plt.figure().clear()

        if (TEST_MODE == 1 or TEST_MODE == 2):
            try:
                bot.sendPhoto(chat_id_user1, photo=open(file_path,'rb'))
                bot.sendPhoto(chat_id_user2, photo=open(file_path,'rb'))
                bot.sendPhoto(chat_id_user3, photo=open(file_path, 'rb'))
            except Exception as e:
                print(e)
        plt.cla()
        plt.clf()
        plt.close('all')
        gc.collect()
        import os
        os.remove(file_path)
        # time2 = tc.take_snapshot()
        # stats = time2.compare_to(time1, 'lineno')
        # print('============')
        # for stat in stats[:3]:
        #     print(stat)
        # print('============')

#    느릴려나?
def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

def myrange(start, end, step):
    r = start
    while (r < end):
        yield r
        r += step

def find_coint_coefficient(pair_df, pair1, pair2, flag): # flag 1이면 CC 계산
    spread_df = pd.DataFrame()
    spread_df['datetime'] = pair_df['datetime']
    # Default Value
    pvalue = 1
    min_cc = 0

    spread_df['spread'] = np.log(pair_df[pair1]) - np.log(pair_df[pair2])
    stable_spread_df = spread_df.copy()
    # print(pair_df)
    if flag == 1:
        for i in myrange(0.5, 2.1, 0.1):
            spread_df['spread'] = np.log(pair_df[pair1]) - i * np.log(pair_df[pair2])
            spread_df2 = spread_df.iloc[:, 1:].values
            # print(spread_df2)
            if pvalue > adfuller(spread_df2, autolag='AIC')[1]:
                pvalue = adfuller(spread_df2, autolag='AIC')[1]
                min_cc = i
                stable_spread_df = spread_df.copy()
            else:
                pvalue = pvalue

    if flag == 0:
        sell_CC = pair_dict[(pair1, pair2)][8]
        spread_df['spread'] = np.log(pair_df[pair1]) - sell_CC * np.log(pair_df[pair2])
        stable_spread_df = spread_df.copy()
        min_cc = sell_CC
        pvalue = adfuller(spread_df.iloc[:,1:].values, autolag='AIC')[1]
    # print("pvalue min = ", pvalue, "CC = ", min_cc)
    return stable_spread_df['spread'], min_cc, pvalue

# compute log spread
def compute_log_spread(pair_df, pair1, pair2, score):
    # spread_df : spread 저장하는 pandas dataframe
    spread_df = pd.DataFrame()
    spread_df['datetime'] = pair_df['datetime']
    spread_df[pair1] = pair_df[pair1]
    spread_df[pair2] = pair_df[pair2]
    spread_df['spread'], CC, pvalue = find_coint_coefficient(pair_df, pair1, pair2, 1)
    # moving average window
    #TODO 뒤에서 몇개 가져올지
    mean = spread_df['spread'].mean()
    std = spread_df['spread'].std()
    std_plus = mean+std
    std_minus = mean-std
    title = re.sub(":","",str(pair_df['datetime'].iloc[0])) + '_' + str(pair1).split('/')[0] + '-' + str(pair2).split('/')[0] + ' SPREAD'
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
            result = coint(S1,S2)
            # result = coint(np.log(S1), np.log(S2))
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            # pvalue_matrix[i, j] = pvalue
            # 0.05 면 충분히 작은가?
            if pvalue < 0.05:
                # pairs.append((keys[i], keys[j]))
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

def select_buy_mvg(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue, leverage, window, buy_sigma ,sigma):
    # action
    # 0 == 아무것도, 1 == 매수, 2 == 수익실현, 3 == 손절,
    # curr_value = pair_df['spread'].iloc[-1]
    # past_value = pair_df['spread'].iloc[-2]
    # print(curr_value)
    # print('==================== select_action ====================')
    # global position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money, pair_dict
    global pair_dict, money, changes, cnt_trade, buy_wait_dict, sell_wait_dict
    # 수익실현
    position = len(pair_dict)
    # print('changes : ', changes, ' position : ', position)
    if position > 4:
        return

    if TEST_MODE == 2:
        try:
            balance = binance.fetch_balance(params={"type": "future"})
            # free	거래에 사용하고 있지 않은 코인양
            # used	거래에 사용하고 있는 코인양
            # total	free + used
            changes = balance['USDT']['free']
        except Exception as e:
            print(e)
            send_message(chat_id_user, str(e))
            return

    this_money = changes / (5.5 - position)
    #
    # this_money_tmp = changes / (10 - position)
    # this_money = np.floor(this_money_tmp * 100 / 100)

    # 매수
    # sigma * std
    curr_zscore = pair_df['zscore_60_5'].iloc[-1]
    if pvalue < 0.05 and curr_zscore > buy_sigma:
        buy_long_symbol = pair2
        buy_short_symbol = pair1
        for pair_ in pair_dict.keys():
            # pair_ = (long, short)
            pair_dict_short_symbol = pair_dict[pair_][1]
            pair_dict_long_symbol = pair_dict[pair_][0]
            if (buy_short_symbol == pair_dict_short_symbol) or (buy_long_symbol == pair_dict_long_symbol) or (buy_long_symbol == pair_dict_short_symbol) or (buy_short_symbol == pair_dict_long_symbol):
                return

        print('+++++++++++++buy+++++++++++++')
        if (TEST_MODE != 3):
            send_message(chat_id_user,'+++++++++++++buy+++++++++++++')

        cnt_trade += 1
        # if TEST_MODE == 1 or TEST_MODE == 2:
        if (TEST_MODE != 3):
            #     지정가 주문
            # orderbook = binance.fetch_order_book('BTC/USDT')
            # # bids 매수 asks 매도 100개 보여줌
            # # orderbook (가격, 수량)
            # print(orderbook['bids'][0][0]) -> 매수에서 가장 높은 가격 -> short 주문 지정가
            # print(orderbook['asks'][0][0]) -> 매도에서 가장 낮은 가격 -> long 주문 지정가
            try:
                long_orderbook = binance.fetch_order_book(buy_long_symbol)
                short_orderbook = binance.fetch_order_book(buy_short_symbol)
                buy_long_price = long_orderbook['bids'][0][0]
                buy_short_price = short_orderbook['asks'][0][0]
                # buy_long_price = long_orderbook['asks'][0][0]
                # buy_short_price = short_orderbook['bids'][0][0]

            except Exception as e:
                print(e)
                send_message(chat_id_user, str(e))
                return
        else:
            buy_long_price = pair_df[buy_long_symbol].iloc[-1]
            buy_short_price = pair_df[buy_short_symbol].iloc[-1]

        # 수량 소수점!?
        buy_long_num = truncate((this_money * CC)/ (CC + 1) / buy_long_price, 4)
        buy_short_num = truncate((this_money) / (CC + 1) / buy_short_price, 4)
        # get_target_price(long, short)
        # binance api 오류 처리
        long_leverage = leverage
        short_leverage = leverage

        # 실제 매매!!!
        # 수량은 소수점 어디까지 되나??
        if TEST_MODE == 2:
            try:
                long_order = binance.create_limit_buy_order(symbol=buy_long_symbol, amount=buy_long_num, price=buy_long_price)
                short_order = binance.create_limit_sell_order(symbol=buy_short_symbol, amount=buy_short_num, price=buy_short_price)
                buy_wait_dict[buy_long_symbol] = (long_order['info']['orderId'], buy_long_num, buy_long_price, buy_long_num * buy_long_price)
                sell_wait_dict[buy_short_symbol] = (short_order['info']['orderId'], buy_short_num, buy_short_price, buy_short_num*buy_short_price)
            #     대기주문 처리해야함
            except Exception as e:
                print(e)
                send_message(chat_id_user, str(e))
                return
        # long_target_l, long_target_s : long coin 매수세 매도세
        # short_target_l, short_target_s : short coin 매수세 매도세
        long_target_l, long_target_s, short_target_l, short_target_s = 0, 0, 0, 0

        # 실제 매매일때는 잔고 긁어오면 됨!!
        if TEST_MODE != 2:
            changes -= buy_long_price * buy_long_num + buy_short_price * buy_short_num

        # position = 1
        # (buy_long_symbol, buy_short_symbol, buy_long_price, buy_short_price, buy_long_num, buy_short_num)
        print('position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, long_leverage, short_leverage, money, changes')
        print(position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num,long_leverage, short_leverage, money, changes)
        print('Show me the money : ', money, ' change : ', changes, ' pvalue = ', pvalue, ' CC = ', CC, ' cnt_trade = ', cnt_trade)
        if (TEST_MODE != 3):
            send_message(chat_id_user, 'position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money, changes')
            send_message(chat_id_user, str(position) + ' ' + str(buy_long_price) + ' ' + str(buy_short_price) + ' ' + str(buy_long_symbol) + ' ' + str(buy_short_symbol) + ' ' + str(buy_long_num) + ' ' + str(buy_short_num) + ' ' + str(money) + ' ' + str(changes))
            send_message(chat_id_user, 'Show me the money : '+str(money)+' change : '+str(changes)+' pvalue = '+str(pvalue)+' CC = '+str(CC)+' cnt_trade = '+str(cnt_trade))

        pair_dict[(pair1, pair2)] = (buy_long_symbol, buy_short_symbol, buy_long_price, buy_short_price, buy_long_num, buy_short_num, long_leverage,short_leverage, CC)
        title = re.sub(":", "", str(pair_df['datetime'].iloc[0])) + '_' + str(pair1).split('/')[0] + '-' + \
                str(pair2).split('/')[0] + '-BUY-WINDOW' + str(window)
        if (TEST_MODE != 3):
            show_plot(pair_df, mean, std_plus, std_minus, title, sigma)

    elif pvalue < 0.05 and curr_zscore < -buy_sigma:
        buy_long_symbol = pair1
        buy_short_symbol = pair2
        for pair_ in pair_dict.keys():
            # pair_ = (long, short)
            pair_dict_short_symbol = pair_dict[pair_][1]
            pair_dict_long_symbol = pair_dict[pair_][0]
            if (buy_short_symbol == pair_dict_short_symbol) or (buy_long_symbol == pair_dict_long_symbol) or (
                    buy_long_symbol == pair_dict_short_symbol) or (buy_short_symbol == pair_dict_long_symbol):
                return

        print('+++++++++++++buy+++++++++++++')
        if (TEST_MODE != 3):
            send_message(chat_id_user, '+++++++++++++buy+++++++++++++')

        cnt_trade += 1
        # if TEST_MODE == 1 or TEST_MODE == 2:
        if (TEST_MODE != 3):
            #     지정가 주문
            # orderbook = binance.fetch_order_book('BTC/USDT')
            # # bids 매수 asks 매도 100개 보여줌
            # # orderbook (가격, 수량)
            # print(orderbook['bids'][0][0]) -> 매수에서 가장 높은 가격 -> short buy 주문 지정가
            # print(orderbook['asks'][0][0]) -> 매도에서 가장 낮은 가격 -> long buy 주문 지정가
            try:
                long_orderbook = binance.fetch_order_book(buy_long_symbol)
                short_orderbook = binance.fetch_order_book(buy_short_symbol)
                # buy_long_price = long_orderbook['asks'][0][0]
                # buy_short_price = short_orderbook['bids'][0][0]
                buy_long_price = long_orderbook['bids'][0][0]
                buy_short_price = short_orderbook['asks'][0][0]
            except Exception as e:
                print(e)
                send_message(chat_id_user, str(e))
                return
        else:
            buy_long_price = pair_df[buy_long_symbol].iloc[-1]
            buy_short_price = pair_df[buy_short_symbol].iloc[-1]

        buy_long_num = truncate((this_money) / (CC + 1) / buy_long_price , 4)
        buy_short_num = truncate((this_money * CC) / (CC + 1) / buy_short_price , 4)
        long_leverage = leverage
        short_leverage = leverage
        # 실제 매매!!!
        # 수량은 소수점 어디까지 되나??
        if TEST_MODE == 2:
            try:
                long_order = binance.create_limit_buy_order(symbol=buy_long_symbol, amount=buy_long_num, price=buy_long_price)
                short_order = binance.create_limit_sell_order(symbol=buy_short_symbol, amount=buy_short_num, price=buy_short_price)
                buy_wait_dict[buy_long_symbol] = (long_order['info']['orderId'], buy_long_num, buy_long_price, buy_long_num * buy_long_price)
                sell_wait_dict[buy_short_symbol] = (short_order['info']['orderId'], buy_short_num, buy_short_price, buy_short_num*buy_short_price)
            #     대기주문 처리해야함
            except Exception as e:
                print(e)
                send_message(chat_id_user, str(e))
                return

        if TEST_MODE != 2:
            changes -= buy_long_price * buy_long_num + buy_short_price * buy_short_num

        # position = 1
        print('position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, long_leverage, short_leverage, money, changes')
        print(position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, long_leverage, short_leverage, money,changes)
        print('Show me the money : ', money, ' change : ', changes, ' pvalue = ', pvalue, ' CC = ', CC, ' cnt_trade = ',cnt_trade)
        if (TEST_MODE != 3):
            send_message(chat_id_user, 'position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money, changes')
            send_message(chat_id_user, str(position) + ' ' + str(buy_long_price) + ' ' + str(buy_short_price) + ' ' + str(buy_long_symbol) + ' ' + str(buy_short_symbol) + ' ' + str(buy_long_num) + ' ' + str(buy_short_num) + ' ' + str(money) + ' ' + str(changes))
            send_message(chat_id_user,'Show me the money : ' + str(money) + ' change : ' + str(changes) + ' pvalue = ' + str(pvalue) + ' CC = ' + str(CC) + ' cnt_trade = ' + str(cnt_trade))

        pair_dict[(pair1, pair2)] = (buy_long_symbol, buy_short_symbol, buy_long_price, buy_short_price, buy_long_num, buy_short_num, long_leverage, short_leverage, CC)
        title = re.sub(":","",str(pair_df['datetime'].iloc[0]))+ '_' + str(pair1).split('/')[0] + '-' + str(pair2).split('/')[0] + '-BUY-WINDOW' + str(window)
        if (TEST_MODE != 3):
            show_plot(pair_df, mean, std_plus, std_minus, title, sigma)


def select_sell_mvg(pair_df, pair1, pair2, mean, std, std_plus, std_minus, pvalue, leverage, window, sigma):
    # position 항상 1
    # action
    # 0 == 아무것도, 1 == 매수, 2 == 수익실현, 3 == 손절,
    # curr_value = pair_df['spread'].iloc[-1]
    # past_value = pair_df['spread'].iloc[-2]
    # print(curr_value)
    # print('==================== select_action ====================')
    # global position, buy_long_price, buy_short_price, buy_long_symbol, buy_short_symbol, buy_long_num, buy_short_num, money, pair_dict
    global pair_dict, money, changes, cnt_good_sell, cnt_10_sell, buy_wait_dict, sell_wait_dict

    # 수익실현
    if TEST_MODE == 2:
        try:
            balance = binance.fetch_balance(params={"type": "future"})
            init_money = balance['USDT']['total']
        except Exception as e:
            print(e)
            send_message(chat_id_user, str(e))
            return

    buy_long_symbol = pair_dict[(pair1, pair2)][0]
    buy_short_symbol = pair_dict[(pair1, pair2)][1]
    buy_long_price = pair_dict[(pair1, pair2)][2]
    buy_short_price = pair_dict[(pair1, pair2)][3]
    buy_long_num = pair_dict[(pair1, pair2)][4]
    buy_short_num = pair_dict[(pair1, pair2)][5]
    long_leverage = pair_dict[(pair1, pair2)][6]
    short_leverage = pair_dict[(pair1, pair2)][7]
    CC = pair_dict[(pair1, pair2)][8]

    # if (mean - past_value) * (mean - curr_value) <= 0:
    curr_zscore = pair_df['zscore_60_5'].iloc[-1]
    if curr_zscore < 0.25 and curr_zscore > -0.25:

        print('+++++++++++++good sell +++++++++++++')
        if (TEST_MODE != 3):
            send_message(chat_id_user, '+++++++++++++good sell +++++++++++++')

        cnt_good_sell += 1

        # 한번 거래하는데 수수료 지정가면 0.02% -> 팔때만 반영 -> 0.04%
        # if TEST_MODE == 1 or TEST_MODE == 2:
        if (TEST_MODE != 3):
            #     지정가 주문
            # orderbook = binance.fetch_order_book('BTC/USDT')
            # # bids 매수 asks 매도 100개 보여줌
            # # orderbook (가격, 수량)
            # print(orderbook['bids'][0][0]) -> 매수에서 가장 높은 가격 -> long sell 주문 지정가
            # print(orderbook['asks'][0][0]) -> 매도에서 가장 낮은 가격 -> short sell 주문 지정가
            try:
                long_orderbook = binance.fetch_order_book(buy_long_symbol)
                short_orderbook = binance.fetch_order_book(buy_short_symbol)
                # sell_long_price = long_orderbook['bids'][0][0]
                # sell_short_price = short_orderbook['asks'][0][0]
                sell_long_price = long_orderbook['asks'][0][0]
                sell_short_price = short_orderbook['bids'][0][0]

                # 수수료로 인해 물량이 바뀔수 있음!!
                # 잔고 조회 할때 symbol에서 USDT 빼야함
                if TEST_MODE == 2:
                    sell_long_num = balance[buy_long_symbol[:-5]]['use']
                    sell_short_num = balance[buy_short_symbol[:-5]]['use']

                    long_order = binance.create_limit_buy_order(symbol=buy_short_symbol, amount=sell_short_num,price=sell_short_price)
                    short_order = binance.create_limit_sell_order(symbol=buy_long_symbol, amount=sell_long_num,price=sell_long_price)
                    buy_wait_dict[buy_short_symbol] = (long_order['info']['orderId'], sell_short_num, sell_short_price, sell_short_num * sell_short_price)
                    sell_wait_dict[buy_long_symbol] = (short_order['info']['orderId'], sell_long_num, sell_long_price, sell_long_num * sell_long_price)

            except Exception as e:
                print(e)
                send_message(chat_id_user, str(e))
                return
        else:
            sell_long_price = pair_df[buy_long_symbol].iloc[-1]
            sell_short_price = pair_df[buy_short_symbol].iloc[-1]

        if TEST_MODE != 2:
            money += -(buy_long_price - sell_long_price) * buy_long_num * long_leverage\
                     + (buy_short_price - sell_short_price) * buy_short_num * short_leverage\
                     - 0.00044 * (buy_long_price * buy_long_num + buy_short_price * buy_short_num) * long_leverage
            changes += buy_long_price * buy_long_num + buy_short_price * buy_short_num\
                       - (buy_long_price - sell_long_price) * buy_long_num * long_leverage \
                       + (buy_short_price - sell_short_price) * buy_short_num * short_leverage \
                        - 0.00044 * (buy_long_price * buy_long_num + buy_short_price * buy_short_num) * long_leverage
            benefit = -(buy_long_price - sell_long_price) * buy_long_num * long_leverage\
                     + (buy_short_price - sell_short_price) * buy_short_num * short_leverage\
                     - 0.00044 * (buy_long_price * buy_long_num + buy_short_price * buy_short_num) * long_leverage
        else:
            try:
                money = balance['USDT']['total']
                changes = balance['USDT']['free']
                # benefit 빼는게 날듯
                benefit = money - init_money
                # 너무 자주 잔고를 보는 건가?
            except Exception as e:
                print(e)
                send_message(chat_id_user, str(e))
                return
        # 수수료 적용
        if benefit <0:
            cnt_10_sell +=1

        print('long 친거',buy_long_symbol,'산 가격',buy_long_price,'판 가격',sell_long_price)
        print('short 친거',buy_short_symbol,'산 가격',buy_short_price,'판 가격',sell_short_price)
        print('Show me the money : ', money , ' change : ', changes , ' pvalue = ', pvalue, ' CC = ', CC, ' benefit = ', benefit, ' 10_sell : ' ,cnt_10_sell, ' good_sell : ', cnt_good_sell)

        if (TEST_MODE != 3):
            send_message(chat_id_user, 'long 친거' + str(buy_long_symbol) + '산 가격' + str(buy_long_price) + '판 가격' + str(sell_long_price))
            send_message(chat_id_user, 'short 친거' + str(buy_short_symbol)+ '산 가격'+ str(buy_short_price) + '판 가격' + str(sell_short_price))
            send_message(chat_id_user, 'Show me the money : ' +str(money)+' change : '+str(changes)+' pvalue = '+str(pvalue)+' CC = '+str(CC)+' benefit = '+str(benefit) + ' 10_sell : '+str(cnt_10_sell)+' good_sell : '+str(cnt_good_sell))

        del pair_dict[(pair1, pair2)]
        title = re.sub(":","",str(pair_df['datetime'].iloc[0])) + '_' + str(pair1).split('/')[0] + '-' + str(pair2).split('/')[0] + '-GOODSELL-WINDOW' + str(window)
        if (TEST_MODE != 3):
            show_plot(pair_df, mean, std_plus, std_minus, title, sigma)

    # 손절 (일단 2 sigma) Warning Value
    # elif (pvalue > 0.05) or (curr_value >= mean + sigma * std or curr_value <= mean - sigma * std):
    # elif (curr_value >= mean + sigma * std or curr_value <= mean - sigma * std):
    # elif (pvalue > 0.5) or (curr_zscore > sigma or curr_zscore < -sigma):
    elif (curr_zscore > sigma or curr_zscore < -sigma):
        print('+++++++++++++10 sell+++++++++++++')
        if (TEST_MODE != 3):
            send_message(chat_id_user, '+++++++++++++10 sell+++++++++++++')
        cnt_10_sell += 1
        # if TEST_MODE == 1 or TEST_MODE == 2:
        if (TEST_MODE != 3):
            #     지정가 주문
            # orderbook = binance.fetch_order_book('BTC/USDT')
            # # bids 매수 asks 매도 100개 보여줌
            # # orderbook (가격, 수량)
            # print(orderbook['bids'][0][0]) -> 매수에서 가장 높은 가격 -> long sell 주문 지정가
            # print(orderbook['asks'][0][0]) -> 매도에서 가장 낮은 가격 -> short sell 주문 지정가
            try:
                long_orderbook = binance.fetch_order_book(buy_long_symbol)
                short_orderbook = binance.fetch_order_book(buy_short_symbol)
                # sell_long_price = long_orderbook['bids'][0][0]
                # sell_short_price = short_orderbook['asks'][0][0]
                sell_long_price = long_orderbook['asks'][0][0]
                sell_short_price = short_orderbook['bids'][0][0]

                if TEST_MODE == 2:
                    # 수수료로 인해 물량이 바뀔수 있음!!
                    # 잔고 조회 할때 symbol에서 USDT 빼야함
                    sell_long_num = balance[buy_long_symbol[:-5]]['use']
                    sell_short_num = balance[buy_short_symbol[:-5]]['use']
                    long_order = binance.create_limit_buy_order(symbol=buy_short_symbol, amount=sell_short_num,price=sell_short_price)
                    short_order = binance.create_limit_sell_order(symbol=buy_long_symbol, amount=sell_long_num,price=sell_long_price)
                    buy_wait_dict[buy_short_symbol] = (long_order['info']['orderId'], sell_short_num, sell_short_price, sell_short_num * sell_short_price)
                    sell_wait_dict[buy_long_symbol] = (short_order['info']['orderId'], sell_long_num, sell_long_price, sell_long_num * sell_long_price)

            except Exception as e:
                print(e)
                send_message(chat_id_user, str(e))
                return
        else:
            sell_long_price = pair_df[buy_long_symbol].iloc[-1]
            sell_short_price = pair_df[buy_short_symbol].iloc[-1]

        if TEST_MODE != 2:
            money += -(buy_long_price - sell_long_price) * buy_long_num * long_leverage\
                     + (buy_short_price - sell_short_price) * buy_short_num * short_leverage \
                     - 0.00044 * (buy_long_price * buy_long_num + buy_short_price * buy_short_num) * long_leverage
            changes += buy_long_price * buy_long_num + buy_short_price * buy_short_num \
                       - (buy_long_price - sell_long_price) * buy_long_num  * long_leverage \
                       + (buy_short_price - sell_short_price) * buy_short_num * short_leverage \
                       - 0.00044 * (buy_long_price * buy_long_num + buy_short_price * buy_short_num) * long_leverage
            benefit = -(buy_long_price - sell_long_price) * buy_long_num * long_leverage\
                     + (buy_short_price - sell_short_price) * buy_short_num * short_leverage\
                     - 0.00044 * (buy_long_price * buy_long_num + buy_short_price * buy_short_num) * long_leverage

        print('long 친거', buy_long_symbol, '산 가격', buy_long_price, '판 가격', sell_long_price)
        print('short 친거', buy_short_symbol, '산 가격', buy_short_price, '판 가격', sell_short_price)
        print('Show me the money : ', money, ' change : ', changes, ' pvalue = ', pvalue, ' CC = ', CC, ' benefit = ', benefit, ' 10_sell : ' ,cnt_10_sell, ' good_sell : ', cnt_good_sell)

        if (TEST_MODE != 3):
            send_message(chat_id_user, 'long 친거' + str(buy_long_symbol) + '산 가격' + str(buy_long_price) + '판 가격' + str(sell_long_price))
            send_message(chat_id_user,'short 친거' + str(buy_short_symbol) + '산 가격' + str(buy_short_price) + '판 가격' + str(sell_short_price))
            send_message(chat_id_user,'Show me the money : ' + str(money) + ' change : ' + str(changes) + ' pvalue = ' + str(pvalue) + ' CC = ' + str(CC) + ' benefit = ' + str(benefit) + ' 10_sell : ' + str(cnt_10_sell) + ' good_sell : ' + str(cnt_good_sell))
        # position = 0
        del pair_dict[(pair1, pair2)]
        title = re.sub(":","",str(pair_df['datetime'].iloc[0])) + '_' + str(pair1).split('/')[0] + '-' + str(pair2).split('/')[0] + '-10SELL-WINDOW' + str(window)
        if (TEST_MODE != 3):
            show_plot(pair_df, mean, std_plus, std_minus, title, sigma)


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
    for since_ in range(start_dt[0], start_dt[1], interval):
        # print('since : ',since)
        if since_ + interval > start_dt[1]:
            print('last since_interval : ', since_ +interval, ' start_dt[1] : ', start_dt[1])
            candle_count = int((start_dt[1] - since_) / (60000 * 5))
        pair_temp = pd.DataFrame()
        for market_key in market_search_space:
            ohlcv = binance.fetch_ohlcv(symbol=market_key, timeframe='5m', limit=candle_count, since=since_)
            df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            # print('df : ',df)
            pair_temp['datetime'] = df['datetime']
            pair_temp[market_key] = df['close']
        pair_df_raw_ = pd.concat([pair_df_raw_, pair_temp], ignore_index=True)
        print(pair_df_raw_)
    return pair_df_raw_

def get_window_pairs(market_search_space, window):
    pair_df_raw_ = pd.DataFrame()
    pair_temp = pd.DataFrame()
    for market_key in market_search_space:
        try:
            ohlcv = binance.fetch_ohlcv(symbol=market_key, timeframe='5m', limit=window)
            df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            # print('df : ',df)
            pair_temp['datetime'] = df['datetime']
            pair_temp[market_key] = df['close']
            del ohlcv,  df

        except Exception as e:
            print(e)
            send_message(chat_id_user, str(e))
            return pair_df_raw_, 1
    pair_df_raw_ = pd.concat([pair_df_raw_, pair_temp], ignore_index=True)
    # print(pair_df_raw_)
    return pair_df_raw_, 0

def get_vaild_keys(sorted_pairs, pair_dict_keys):
    temp = []
    for i in range(0, len(sorted_pairs) - 1, 1):
        temp.append(sorted_pairs[i][0])
    # print("temp : " ,temp)
    # print("pair_dict_keys : ", pair_dict_keys)
    valid_keys = []
    if len(pair_dict_keys) == 0:
        for i in temp:
            # if ((i[0],i[1]) not in pair_dict_keys) and ((i[1],i[0]) not in pair_dict_keys):
            valid_keys.append(i)
        # print("valid_keys : ",valid_keys)
    else:
        for i in temp:
            # if ((i[0],i[1]) not in pair_dict_keys) and ((i[1],i[0]) not in pair_dict_keys):
            for pair_dict_ in pair_dict_keys:
                if ((i[0] != pair_dict_[0]) and (i[0] != pair_dict_[1])) and ((i[1] != pair_dict_[0]) and (i[1] != pair_dict_[1])):
                    valid_keys.append(i)
        # print("valid_keys : ",valid_keys)
    return valid_keys

# 변동성 돌파 전략
def get_target_price(long_symbol, short_symbol, time):
    # print(long_symbol)
    # print(short_symbol)
    global binance
    # time.day 가 -1
    time = time - pd.DateOffset(days=1)
    # print(time.year)
    # print(time.month)
    # print(time.day)
    if time.month < 10:
        month = str(0) + str(time.month)
    else:
        month = str(time.month)
    if time.day < 10:
        day = str(0) + str(time.day)
    else:
        day = str(time.day)
    start_dt_ = datetime.strptime(str(time.year)+month+day, "%Y%m%d")
    # print(start_dt_)
    start_dt = calendar.timegm(start_dt_.utctimetuple()) * 1000
    # print(start_dt)
    # 12시 기준

    long_coin = binance.fetch_ohlcv(symbol=long_symbol,timeframe='1d',since=start_dt,limit=2)
    short_coin = binance.fetch_ohlcv(symbol=short_symbol, timeframe='1d', since=start_dt, limit=2)
    # print(long_coin)
    # print(short_coin)
    long_df = pd.DataFrame(long_coin, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    long_df['datetime'] = pd.to_datetime(long_df['datetime'], unit='ms')
    short_df = pd.DataFrame(short_coin, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    short_df['datetime'] = pd.to_datetime(short_df['datetime'], unit='ms')
    # print('long df', long_df)
    # print('short_df',short_df)
    long_yesterday = long_df.iloc[-2]
    long_today = long_df.iloc[-1]
    short_yesterday = short_df.iloc[-2]
    short_today = short_df.iloc[-1]

    long_target_l = long_today['open'] + (long_yesterday['high'] - long_yesterday['low']) * 0.5
    long_target_s = long_today['open'] - (long_yesterday['high'] - long_yesterday['low']) * 0.5

    short_target_l = short_today['open'] + (short_yesterday['high'] - short_yesterday['low']) * 0.5
    short_target_s = short_today['open'] - (short_yesterday['high'] - short_yesterday['low']) * 0.5
    # print(long_target_l, long_target_s, short_target_l, short_target_s)
    return long_target_l, long_target_s, short_target_l, short_target_s


def position_check(curr_position, pair1, pair2):
    global pair_dict, buy_wait_dict, sell_wait_dict

    result_pos = False
    pair1_key = pair1[:-5] + pair1[-4:]
    pair2_key = pair2[:-5] + pair2[-4:]  # romove '/' <= x/usdt
    long_symbol = pair_dict[(pair1, pair2)][0]
    pair1_position = 1 if (long_symbol == pair1) else 0
    pair2_position = 1 if (long_symbol == pair2) else 0

    pair1_open_orders = False
    pair2_open_orders = False

    if pair1_key in curr_position:
        open_orders1 = binance.fetch_open_orders(symbol=pair1)
        if open_orders1 is not None:
            pair1_open_orders = True
            pair1_remaining = open_orders1['remaining']
        else:
            pair1_open_orders = False

    if pair2_key in curr_position:
        open_orders2 = binance.fetch_open_orders(symbol=pair2)
        if open_orders2 is not None:
            pair2_open_orders = True
            pair2_remaining = open_orders2['remaining']
        else:
            pair2_open_orders = False

    if pair1_key in curr_position and pair1_open_orders == False:
        print("pair1 position exist")
        result_pos = True
    else:
        # 시장가 주문
        # pair_dict[(pair1, pair2)][]
        # pair1 is long
        if pair1_position:
            buy_long_num = pair_dict[(pair1, pair2)][4]
            buy_long_price = pair_dict[(pair1, pair2)][2]
            order_id = buy_wait_dict[pair1][0]
            resp = binance.cancel_order(id=order_id, symbol=pair1)
            # 이전 돈
            if pair1_open_orders:
                last_money = pair1_remaining * buy_long_price
            else:
                last_money = buy_long_num * buy_long_price
            # 현재가격
            curr_price = binance.fetch_ticker(pair1)
            curr_num = truncate(last_money / curr_price, 4)
            order = binance.create_market_buy_order(symbol=pair1, amount=curr_num)
            pair_dict[(pair1, pair2)][4] = curr_num
            del buy_wait_dict[pair1]
            result_pos = False
        # pair1 is short
        else:
            buy_short_num = pair_dict[(pair1, pair2)][5]
            buy_short_price = pair_dict[(pair1, pair2)][3]
            order_id = sell_wait_dict[pair1][0]
            resp = binance.cancel_order(id=order_id, symbol=pair1)
            # 이전 돈
            if pair1_open_orders:
                last_money = pair1_remaining * buy_short_price
            else:
                last_money = buy_short_num * buy_short_price
            # 현재가격
            curr_price = binance.fetch_ticker(pair1)
            curr_num = truncate(last_money / curr_price, 4)
            order = binance.create_market_sell_order(symbol=pair1, amount=curr_num)
            pair_dict[(pair1, pair2)][5] = curr_num
            del sell_wait_dict[pair1]
            result_pos = False

    if pair2_key in curr_position and pair2_open_orders == False:
        print("pair2 position exist")
        result_pos = True
    else:
        # 시장가 주문
        # pair_dict[(pair1, pair2)][]
        # pair1 is long ? -> long 주문
        # pair2 is short? -> short 주문
        # pair2 is long
        if pair2_position:
            buy_long_num = pair_dict[(pair1, pair2)][4]
            buy_long_price = pair_dict[(pair1, pair2)][2]
            order_id = buy_wait_dict[pair2][0]
            resp = binance.cancel_order(id=order_id, symbol=pair2)
            # 이전 돈
            if pair2_open_orders:
                last_money = pair2_remaining * buy_long_price
            else:
                last_money = buy_long_num * buy_long_price
            # 현재가격
            curr_price = binance.fetch_ticker(pair2)
            curr_num = truncate(last_money / curr_price, 4)
            order = binance.create_market_buy_order(symbol=pair2, amount=curr_num)
            pair_dict[(pair1, pair2)][4] = curr_num
            del buy_wait_dict[pair1]
            result_pos = False
        # pair2 is short
        else:
            buy_short_num = pair_dict[(pair1, pair2)][5]
            buy_short_price = pair_dict[(pair1, pair2)][3]
            order_id = sell_wait_dict[pair2][0]
            resp = binance.cancel_order(id=order_id, symbol=pair2)
            # 이전 돈
            if pair2_open_orders:
                last_money = pair2_remaining * buy_short_price
            else:
                last_money = buy_short_num * buy_short_price
            # 현재가격
            curr_price = binance.fetch_ticker(pair2)
            curr_num = truncate(last_money / curr_price, 4)
            order = binance.create_market_sell_order(symbol=pair2, amount=curr_num)
            pair_dict[(pair1, pair2)][5] = curr_num
            del sell_wait_dict[pair1]
            result_pos = False

    return result_pos


# leverage = 배율
# candle_count = 5분봉 몇개
# money
def back_test(market_search_space, money, leverage, candle_count, window, buy_sigma ,sigma,since=None):
    print('********* [LABISS Backtest] *********')
    print('현재 잔고 : ', money, 'USDT, leverage : x', leverage, ' Number of Candle : ', candle_count, ' window : ', window, ' buy_sigma : ' ,buy_sigma,'sigma :' ,sigma,' Target test : ', TARGET_MODE)
    pair_df = pd.DataFrame()
    pair_df_raw = pd.DataFrame()

    # backdata 가져오기
    # TODO while문에 try 되게끔
    try:
        pair_df_raw_ = get_backdata_candle(since, candle_count, market_search_space)
        print('pair_df_raw_',pair_df_raw_)
    # except binance.exceptions as e:
    except Exception as e:
        print(e)
        pair_df_raw_ = pd.DataFrame()
        # 5분 sleep
        time.sleep(1*60)
    # pair 찾기
    init_money = money
    # window 이동시키면서 누적으로도 테스트 해봐야함
    candle_count = len(pair_df_raw_.index)
    # binance api error났을 때 종료
    if candle_count == 0:
        return
    # 5분 마다 이동시키면서
    for i in range(0, candle_count-window):
        # memory trace
        # time1 = tc.take_snapshot()
        print('step : ', i, '/', candle_count-window)
        # 현재 포지션에 대해서 select action
        for (pair1, pair2) in list(pair_dict.keys()):
            # print("pair1 : ", pair1, " pair2: ", pair2)
            pair_df_raw['datetime'] = pair_df_raw_['datetime']
            pair_df_raw[pair1] = pair_df_raw_[pair1]
            pair_df_raw[pair2] = pair_df_raw_[pair2]
            pair_df = pair_df_raw.iloc[i:i + window].copy()
            pair_df['spread'], CC, pvalue = find_coint_coefficient(pair_df, pair1, pair2, 0)
            mean = pair_df['spread'].mean()
            std = pair_df['spread'].std()
            std_plus = mean + std
            std_minus = mean - std
            # print(pair_df)
            # 팔지 결정
            if MVG == 1:
                pair_df['ratios_mavg5'] = pair_df['spread'].rolling(window=5, center=False).mean()
                pair_df['ratios_mavg60'] = pair_df['spread'].rolling(window=60, center=False).mean()
                pair_df['std'] = pair_df['spread'].rolling(window=60, center=False).std()
                pair_df['zscore_60_5'] = (pair_df['ratios_mavg5'] - pair_df['ratios_mavg60']) / pair_df['std']

            if MVG == 1:
                select_sell_mvg(pair_df, pair1, pair2, mean, std, std_plus, std_minus, pvalue, leverage, window, sigma)
            # else:
            #     select_sell(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue, leverage, window, sigma)

            # show_plot(pair_df, mean, std_plus, std_minus, title)
            # print('Show me the money : ', money)

        pair_number = len(pair_dict)
        print("pair_number", pair_number)
        print("pair_dict : ",pair_dict)
        if pair_number < 5:
            # 먼저 pair_dict key에 있는 페어로 select action
            # 끝나고 새로운 페어 찾기
            # pair_df, pair1, pair2 = find_pair()
            pair_df_ = pair_df_raw_.iloc[i:i+window].copy()

            scores, pvalues, pairs = find_cointegrated_pairs(pair_df_.iloc[:, 1:14])
            sorted_pairs = sorted(pvalues.items(), key=operator.itemgetter(1))
            # print('sorted_pairs len : ', len(sorted_pairs))
            valid_keys = get_vaild_keys(sorted_pairs, list(pair_dict.keys()))
            for (pair1, pair2) in valid_keys:
                # print("pair1 : ", pair1, " pair2: ", pair2)
                pair_df_raw['datetime'] = pair_df_raw_['datetime']
                pair_df_raw[pair1] = pair_df_raw_[pair1]
                pair_df_raw[pair2] = pair_df_raw_[pair2]
                pair_df = pair_df_raw.iloc[i:i + window].copy()
                pair_df['spread'], CC, pvalue = find_coint_coefficient(pair_df, pair1, pair2, 1)
                mean = pair_df['spread'].mean()
                std = pair_df['spread'].std()
                std_plus = mean + std
                std_minus = mean - std
                if MVG == 1:
                    pair_df['ratios_mavg5'] = pair_df['spread'].rolling(window=5, center=False).mean()
                    pair_df['ratios_mavg60'] = pair_df['spread'].rolling(window=60, center=False).mean()
                    pair_df['std'] = pair_df['spread'].rolling(window=60, center=False).std()
                    pair_df['zscore_60_5'] = (pair_df['ratios_mavg5'] - pair_df['ratios_mavg60']) / pair_df['std']

                title = "test"
                # print(pair_df)
                if MVG == 1:
                    select_buy_mvg(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue, leverage, window, buy_sigma, sigma)
                # else:
                #     select_buy(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue, leverage, window, buy_sigma, sigma)

                # show_plot(pair_df, mean, std_plus, std_minus, title)
        # print('Show me the money : ', money)

        # memory trace
        # time2 = tc.take_snapshot()
        # stats = time2.compare_to(time1, 'lineno')
        # print('============')
        # for stat in stats[:3]:
        #     print(stat)
        # print('============')

# leverage = 배율
# candle_count = 5분봉 몇개
# money
def labiss_main(market_search_space, leverage, window, buy_sigma, sigma):
    # memory trace
    time1 = tc.take_snapshot()
    print('********* [LABISS Active] *********')
    global money, pair_dict, buy_wait_dict, sell_wait_dict
    # 현재 잔고 조회
    if TEST_MODE == 2:
        try:
            balance = binance.fetch_balance(params={"type": "future"})
            money = balance['USDT']['total']
            positions = balance['info']['positions']
            curr_position = []
            for position in positions:
                curr_position.append(position["symbol"])

            # 이게 안되면 포지션 별로 잔고 긁어와야함
        except Exception as e:
            print(e)
            send_message(chat_id_user, str(e))
            return

    print('현재 잔고 : ', money, 'USDT, leverage : x', leverage, ' window : ', window, ' buy_sigma : ' ,buy_sigma,' sigma :' ,sigma)
    send_message(chat_id_user, '********* [LABISS Active] *********')
    if TEST_MODE == 2:
        send_message(chat_id_user, '현재 잔고 : '+str(money))

    for (pair1, pair2) in list(pair_dict.keys()):
        if TEST_MODE == 2:
            is_pos = position_check(curr_position, pair1, pair2)
            if is_pos == False:
                print("curr position check failed, next position check start")
                pass

        pair_df = pd.DataFrame()
        # binance error 날 경우 5분뒤 다시 시도 하게끔
        try:
            ohlcv_pair1 = binance.fetch_ohlcv(symbol=pair1, timeframe='5m', limit=window)
            ohlcv_pair2 = binance.fetch_ohlcv(symbol=pair2, timeframe='5m', limit=window)
            df_pair1 = pd.DataFrame(ohlcv_pair1, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df_pair2 = pd.DataFrame(ohlcv_pair2, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            pair_df['datetime'] = pd.to_datetime(df_pair1['datetime'], unit='ms')
            pair_df[pair1] = df_pair1['close']
            pair_df[pair2] = df_pair2['close']
            # print('in pair',pair_df)
            pair_df['spread'], CC, pvalue = find_coint_coefficient(pair_df, pair1, pair2, 0)
            mean = pair_df['spread'].mean()
            std = pair_df['spread'].std()
            std_plus = mean + std
            std_minus = mean - std
            # print(pair_df)
            # 팔지 결정
            if MVG == 1:
                pair_df['ratios_mavg5'] = pair_df['spread'].rolling(window=5, center=False).mean()
                pair_df['ratios_mavg60'] = pair_df['spread'].rolling(window=60, center=False).mean()
                pair_df['std'] = pair_df['spread'].rolling(window=60, center=False).std()
                pair_df['zscore_60_5'] = (pair_df['ratios_mavg5'] - pair_df['ratios_mavg60']) / pair_df['std']

            if MVG == 1:
                select_sell_mvg(pair_df, pair1, pair2, mean, std, std_plus, std_minus, pvalue, leverage, window, sigma)
            # select_sell(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue, leverage, window, sigma)
            # show_plot(pair_df, mean, std_plus, std_minus, title)
            # print('Show me the money : ', money)
            del ohlcv_pair1, ohlcv_pair2, df_pair1, df_pair2

        except Exception as e:
            print(e)
            send_message(chat_id_user, str(e))
            return

    pair_number = len(pair_dict)
    print("pair_number", pair_number)
    print("pair_dict : ",pair_dict)
    send_message(chat_id_user, 'pair_number ' + str(pair_number))
    # dict to string
    if pair_number < 5:
        # 먼저 pair_dict key에 있는 페어로 select action
        # 끝나고 새로운 페어 찾기
        # pair_df, pair1, pair2 = find_pair()
        try:
            pair_df_, window_error = get_window_pairs(market_search_space, window)
            if window_error == 1:
                return
            scores, pvalues, pairs = find_cointegrated_pairs(pair_df_.iloc[:, 1:14])
            sorted_pairs = sorted(pvalues.items(), key=operator.itemgetter(1))
            valid_keys = get_vaild_keys(sorted_pairs, list(pair_dict.keys()))
            for (pair1, pair2) in valid_keys:
                pair_df = pd.DataFrame()
                pair_df['datetime'] = pair_df_['datetime']
                pair_df[pair1] = pair_df_[pair1]
                pair_df[pair2] = pair_df_[pair2]
                pair_df['spread'], CC, pvalue = find_coint_coefficient(pair_df, pair1, pair2, 1)
                mean = pair_df['spread'].mean()
                std = pair_df['spread'].std()
                std_plus = mean + std
                std_minus = mean - std
                title = "test"
                # print(pair_df)
                if MVG == 1:
                    pair_df['ratios_mavg5'] = pair_df['spread'].rolling(window=5, center=False).mean()
                    pair_df['ratios_mavg60'] = pair_df['spread'].rolling(window=60, center=False).mean()
                    pair_df['std'] = pair_df['spread'].rolling(window=60, center=False).std()
                    pair_df['zscore_60_5'] = (pair_df['ratios_mavg5'] - pair_df['ratios_mavg60']) / pair_df['std']

                if MVG == 1:
                    select_buy_mvg(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue, leverage, window, buy_sigma, sigma)
                # select_buy(pair_df, pair1, pair2, mean, std, std_plus, std_minus, CC, pvalue, leverage, window, buy_sigma,sigma)
                # show_plot(pair_df, mean, std_plus, std_minus, title)
            del scores, pvalues, pairs
        except Exception as e:
            print(e)
            send_message(chat_id_user, str(e))
            return
    # print('Show me the money : ', money)
    send_message(chat_id_user, '********* [LABISS Deactive] *********')
    gc.collect()
    del pair_df, pair_df_

    # memory trace
    time2 = tc.take_snapshot()
    stats = time2.compare_to(time1, 'lineno')
    print('============')
    for stat in stats[:3]:
        print(stat)
    # print('============')

if __name__ == '__main__':
    market_search_space = ['ETH/USDT', 'BTC/USDT', 'XRP/USDT', 'ADA/USDT', 'LINK/USDT', 'VET/USDT', 'EOS/USDT',
                           'LTC/USDT', 'ETC/USDT','BCH/USDT', 'TRX/USDT', 'XLM/USDT']
    if TEST_MODE == 2:
        # 추가 해야함
        api_key = ""
        secret = ""

        binance = ccxt.binance(config={
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })

        # leverage 설정
        for symbol in market_search_space:
            market = binance.market(symbol)
            leverage = 5

            resp = binance.fapiPrivate_post_leverage({
                'symbol': market['id'],
                'leverage': leverage
            })
    else:
        binance = ccxt.binance()

    if (TEST_MODE != 3):
        money = 100
        window = 300
        buy_sigma = 1.5
        sell_sigma = 3.5
        changes = 100
        pair_dict = {}
        buy_wait_dict = {}
        sell_wait_dict = {}
        cnt_trade = 0
        cnt_good_sell = 0
        cnt_10_sell = 0
        # market_search(market_search_space)
        try:
            labiss_main(market_search_space, 5, window, buy_sigma, sell_sigma)
        except Exception as e:
            print(e)
            # pair_df_raw_ = pd.DataFrame()
            # # 5분 sleep
            # time.sleep(1 * 60)
            send_message(chat_id_user, str(e))
        sch.every(5).minutes.do(labiss_main, market_search_space, 5, window, buy_sigma ,sell_sigma)

        while True:
            sch.run_pending()
            # time.sleep(0.5)
    else:
        # money, leverage, candle_count, window
        date = [['20210101', '20210109']]
        # date = [['20210129', '20210222']]
        # date = [['20210301', '20210314']]
        # date = [['20210326', '20210331'], ['20210410', '20210414'], ['20210426', '20210502']]
        # date = [['20210326', '20210331']]
        # date = [['20210410', '20210414']]
        # date = ['20210426', '20210502']
        # date = [['20210301', '20210314'], ['20210326', '20210331']]
        # date = [['20210410', '20210414'], ['20210426', '20210502']]
        # start_window = 0
        # end_window = 300
        # window = [50, 100, 500]
        window = [100, 300, 500]
        # window = [300]
        sell_sigma = 3.5
        buy_sigma = [1, 1.5, 2, 2.5]
        # buy_sigma = [1.5]
        for date_ in date:
            for buy_sigma_ in buy_sigma:
                for window_ in window:
                    print("START BACK_TEST!!!")
                    money = 100
                    changes = 100
                    pair_dict = {}
                    cnt_trade = 0
                    cnt_good_sell = 0
                    cnt_10_sell = 0
                    back_test(market_search_space, money, 5, 1000, window_, buy_sigma_ , sell_sigma ,date_)