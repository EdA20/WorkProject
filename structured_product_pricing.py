import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm.auto import tqdm
import pyodbc
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=*******;"
                      "Database=*******;"
                      "Trusted_Connection=yes;")

''' 
Данная функция подсчитывает потенциальную выплату по структурным продуктам
с эффектом памяти
Вх. данные - массив с выплатами фикс. купонов, напр. [0, 200, 0, 0, 200]
Вых. данные - массив с выплатми с применением эффекта памяти,  [0, 400, 0, 0, 600]
'''


def memory_effect(x: list) -> list:
    j = 1
    for k, i in enumerate(x):
        if i == 0:
            j += 1
        else:
            x[k] = j * i
            j = 1
    return x


'''
Геометрические броуновское движение.
n - кол-во дат наблюдений
mu - дрифт базового актива (исп-ся безрисковая ставка в соответствующей валюте)
sigma - волатильность (напр. исторически посчитанная за 120 дней)
dt - шаг, с которым строится 1 ценовой путь (зачастую используется время до даты наблюдения, чтобы не генерировать лишние набл.)
S0 - цена базовых активов в начальный момент
option_type - тип опциона по кол-ву базовых активов (1 базовый актив или несколько)
chol - разложение Холнцки (нужна при генерации скоррелированных базовых активов)
n_path - кол-во путей/вариантов движения базового актива

В данной функции реализован один из простейщих методов снижения дисперсии - antithetic paths.

На выходе мы получаем два 3-мерных массива, где 
по нулевому индексу размерности - кол-во путей
по первому индекса размерности - кол-во дат наблюдений
по второму индексу размерности - кол-во базовых активов
'''
def brownian_motion3(n, mu, sigma, dt, S0, option_type, chol, n_path=90000):
    rand = np.float32(np.random.normal(size=(n_path, n, len(sigma))))

    if 'phoenix' in option_type or 'weighted' in option_type:
        x = np.exp(
            (mu - sigma ** 2 / 2) * dt
            + sigma * (rand @ chol.T) * dt ** (1 / 2)
        )

        y = np.exp(
            (mu - sigma ** 2 / 2) * dt
            + sigma * -(rand @ chol.T) * dt ** (1 / 2)
        )
    else:
        x = np.exp(
            (mu - sigma ** 2 / 2) * dt
            + sigma * rand * dt ** (1 / 2)
        )

        y = np.exp(
            (mu - sigma ** 2 / 2) * dt
            + sigma * -rand * dt ** (1 / 2)
        )

    x = np.concatenate([np.ones((n_path, 1, len(sigma)), dtype=np.float32), x], axis=1)
    y = np.concatenate([np.ones((n_path, 1, len(sigma)), dtype=np.float32), y], axis=1)

    x = S0 * x.cumprod(axis=1)
    y = S0 * y.cumprod(axis=1)

    return x, y

'''
Подсчет цены и греков колл опциона по формуле БШ
'''
class Call:

    def __init__(self, S0, strike, sigma, rate, amount, t, **kwargs):
        self.S0 = S0
        self.strike = strike
        self.sigma = sigma
        self.rate = rate
        self.amount = amount
        self.t = t
        self.d1 = (np.log(self.S0 / self.strike) + (self.rate + self.sigma ** 2 / 2) * self.t) / (
                    self.sigma * np.sqrt(self.t))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.t)

    def price(self):
        call_pv = norm.cdf(self.d1) * self.S0 - norm.cdf(self.d2) * self.strike * np.exp(-self.rate * self.t)
        return call_pv * self.amount

    def delta(self):
        return norm.cdf(self.d1) * self.amount

    def vega(self):
        return self.S0 * np.sqrt(self.t) * norm.pdf(self.d1) * self.amount

    def gamma(self):
        return norm.pdf(self.d1) / (self.S0 * self.sigma * np.sqrt(self.t)) * self.amount


'''
Подсчет справедливой стоиомости опциона 
val_date - дата оценки
notes - исины оцениваемых нот
var_report_date - указать дату для подсчета цен, которые будут использоваться для VaR
stress - указать True для подсчета стресс-сценария 
'''


def option_fair_value(val_date: pd.Datetime([]),
                      notes=note_isins,
                      var_report_date=False,
                      stress=False):

    if not var_report_date:
        var_report_date = val_date

    info = pd.read_sql_query('select * from structured_products', cnxn, index_col='ISIN')

    prices_all = pd.read_sql_query('''select * 
                              from structured_products_prices_new''',
                                   cnxn)
    prices_all = prices_all.pivot(index='date', columns='ISIN', values='price').rename_axis(None,
                                                                                            axis="columns").reset_index()
    prices_all = prices_all.fillna(method='bfill')
    prices_all = prices_all.sort_values('date', ascending=False, ignore_index=True)

    isins = pd.read_sql_query('''select * 
                              from structured_products_isins''',
                              cnxn, index_col='ISIN_note')
    df1 = pd.read_excel('rates.xlsx', usecols=[3, 4]).set_axis(['index', 'value'], axis=1)
    dct = {'SONIA': 'gbx', 'HONIA': 'hkd', 'SARON': 'chf', 'ESTR (€STR)': 'eur', 'RUONIA rate': 'rub', 'SOFR': 'usd'}
    df1['currency'] = df1['index'].map(dct)
    rate = df1[df1['currency'] == 'rub']['value'].values / 100

    all_option_prices = {}
    for isin in tqdm(notes):

        isin_underlying = isins.loc[[isin]]['ISIN_under'].values
        prices = prices_all[np.insert(isin_underlying, 0, 'date')]
        initial = prices[prices['date'] == info['initial_date'][isin]].iloc[:, 1:].values

        option_type = info.loc[isin, 'type']

        if 'part' in option_type:
            barier_date = pd.read_sql_query('''select * 
                                      from structured_products_part''',
                                            cnxn, index_col='ISIN').loc[isin]
        else:
            basket = pd.read_sql_query('''select * 
                                      from structured_products_phoenix''',
                                       cnxn, index_col='ISIN').loc[isin]
            barier_date = basket['barrier_date'].copy()
            barier_value = basket['barrier'].copy()

        curr_idx = np.where(isins.loc[[isin]]['currency'].values[:, None] == df1['currency'].values)[1]
        rate_curr = df1.loc[curr_idx, 'value'].values / 100

        if barier_date[-1] < val_date:
            continue

        ret = pd.concat([
            prices[['date']],
            pd.DataFrame(np.log(prices.iloc[:-1, 1:].values / prices.iloc[1:, 1:].values), columns=prices.columns[1:])],
            axis=1).dropna()

        idx = ((ret['date'] - val_date).astype('timedelta64[D]').astype(int)).values
        idx = np.argmin(np.where(idx > 0, np.inf, abs(idx)))
        ret_sliced = ret.iloc[idx:idx + 120]

        params = {
            'mu': np.float32(rate_curr),  # дрифт (исп-ся безрисковая ставка в $)
            'sigma': np.float32(ret_sliced.drop('date', axis=1).std(ddof=0) * 250 ** (1 / 2)),  # ст.откл.
            'dt': np.float32(1 / 52),  # шаг для прогнозирования цены (день, неделя, месяц)
            'S0': np.float32(prices[prices['date'] == val_date].iloc[:, 1:].values), # цена, начиная с которой строится прогноз
            'option_type': option_type,  # тип опциона
            'n_path': 120000
        }

        if len(params['S0']) == 0:
            idx2 = ((prices['date'] - val_date).astype('timedelta64[D]').astype(int)).values
            idx2 = np.argmin(np.where(idx > 0, np.inf, abs(idx)))
            params['S0'] = np.float32(prices.iloc[idx2, 1:].values)

        delt = ((barier_date - var_report_date).dt.days / 365).values
        if len(delt) > 1:
            delt = delt - np.roll(np.where(delt > 0, delt, 0), 1)
            delt = np.where(delt > 0, delt, 0)

        coupon = np.ones(len(barier_date)) * info['nominal'][isin] * info['coupon/part_rate'][isin]
        discount = np.exp(delt * -rate)
        params['chol'] = np.float32(np.linalg.cholesky(ret_sliced.corr()))

        params['dt'] = delt[np.argwhere(delt > 0)]
        params['n'] = len(delt[np.argwhere(delt > 0)])

        coefs = [1]

        if stress:
            if isin != 'RU000A1020E0':
                coefs = [1, 1.1, 0.8, 0.96, 1.07]
            else:
                coefs = [1, 0.85, 0.89, 0.91, 0.63]

        option_price = []
        params_old = params.copy()
        np.random.seed(1)

        for coef in coefs:

            if stress:
                params['S0'] = params_old['S0'] * coef

            if option_type == 'phoenix':

                his = (prices[np.isin(prices['date'], barier_date[delt == 0])].sort_values('date', ascending=True).iloc[
                       :, 1:].values / initial >
                       barier_value[delt == 0].values[:, None]).all(axis=1)

                x, y = brownian_motion3(**params)

                pay_x = (x[:, 1:] / initial > barier_value[delt > 0].values[:, None]).all(axis=2)
                pay_y = (y[:, 1:] / initial > barier_value[delt > 0].values[:, None]).all(axis=2)

                c_x = np.hstack([np.tile(his, (params['n_path'], 1)), pay_x]) * coupon
                c_y = np.hstack([np.tile(his, (params['n_path'], 1)), pay_y]) * coupon
                for i in range(len(c_x)):
                    c_x[i] = memory_effect(c_x[i])
                    c_y[i] = memory_effect(c_y[i])

                value_x = (c_x * discount)[:, delt > 0].sum(axis=1).mean()
                value_y = (c_y * discount)[:, delt > 0].sum(axis=1).mean()

                option_price.append((value_x + value_y) / 2)

            else:

                x, y = brownian_motion3(**params)

                perform_x = ((x[:, -1] / initial).mean(axis=1) - 1) * coupon
                perform_y = ((y[:, -1] / initial).mean(axis=1) - 1) * coupon

                pay_x = np.where(perform_x > 0, perform_x, 0)
                pay_y = np.where(perform_y > 0, perform_y, 0)

                value_x = (pay_x * discount).mean()
                value_y = (pay_y * discount).mean()

                option_price.append((value_x + value_y) / 2)

        all_option_prices[isin] = option_price

    if stress:
        columns = ['TCC', 'Умеренный сценарий кризис', 'Умеренный сценарий восстановление',
                   'Прогнозный cценарий', 'Пессимистичный сценарий кризис']

        df = -pd.DataFrame(all_option_prices.values(), index=all_option_prices.keys(), columns=columns)
        df = df.astype(int)
        df.index.name = 'Security code'

    else:
        df = pd.DataFrame(all_option_prices).T.reset_index().set_axis(['ISIN', 'TCC'], axis=1)
        df['TCC'] = df['TCC'].astype(int)

    return df