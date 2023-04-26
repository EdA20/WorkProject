class Bond():
    '''
    Для инициализации экземпляра класса бонд необходимо передать след. параметры:

        * market_price - рыночная стоимость
        * coupon - купонная ставка
        * today - даты, на кот-ую расчитывается модельная цена
        * maturity - дата погашения
        * coupon_date - даты выплат купона

    Доп. параметры, содержащие дефолтные значения (их можно менять)

        * face_value - номинал (дефолт = 1000)
        * nkd - НКД (дефолт = 0)
        * oferta - кол-во лет до оферты (дефолт = False)
        * t1 = '2022-01-01' - граница даты для sql таблица по GCurve
        * t2 = '2023-01-01' - граница даты для sql таблица по GCurve

    '''

    def __init__(self, maturity_info, coupon_info, market_info,
                 face_value=100, oferta=False, amort=False, t1='2022-01-01', t2='2023-01-01'):

        self.today = np.array(market_info['Date'], dtype='datetime64[D]', ndmin=2).T
        self.maturity = np.array(maturity_info.index, dtype='datetime64[D]')
        self.coupon_date = np.array(coupon_info['BeginPeriod'].values, dtype='datetime64[D]')
        self.coupon_end_date = np.array(coupon_info['EndPeriod'].values, dtype='datetime64[D]')
        self.t1 = t1
        self.t2 = t2
        self.market_price = market_info['Price'].values
        shift = (self.coupon_date[1] - self.coupon_end_date[0]).astype(int)

        if oferta:
            self.maturity = np.array(oferta, dtype='datetime64[D]')
            self.coupon_end_date = self.coupon_end_date[self.coupon_end_date <= self.maturity]

        # подсчет купона
        self.coupon = (coupon_info['CouponRate'].values *
                       ((self.coupon_end_date - self.coupon_date).astype(int) + shift) / 365)

        # подсчет номинала
        self.nominal = (((self.maturity - self.today).astype(int) > 0) * maturity_info['MtyPart'].values).sum(axis=1)
        self.face_value = maturity_info['MtyPart'].values

    def __init__(self, market_price, coupon, today, maturity, coupon_date,
                 face_value=1000, nkd=False, oferta=False, t1='2022-01-01', t2='2023-01-01'):

        self.market_price = np.array(market_price)
        self.face_value = face_value
        if all(nkd) == False:
            self.nkd = np.array([0 for i in range(len(today))])
        else:
            self.nkd = nkd

        self.today = np.array(today, dtype='datetime64[D]', ndmin=2).T
        self.maturity = np.array(maturity, dtype='datetime64[D]')
        self.coupon_date = np.array(coupon_date, dtype='datetime64[D]')
        self.t1 = t1
        self.t2 = t2

        if oferta:
            self.maturity = np.array(oferta, dtype='datetime64[D]')
            self.coupon_date = self.coupon_date[self.coupon_date <= self.maturity]

        self.coupon = np.array([face_value * coupon for i in range(len(self.coupon_date))])

    def nkd(self):
        delta_total = (self.today - self.coupon_date).astype(int)
        delta = np.where(delta_total >= 0, delta_total, np.inf)
        days = delta.min(axis=1)
        days_total = np.array((b.coupon_end_date - b.coupon_date), dtype='timedelta64[D]').astype(int)
        nkd = days / days_total[delta.argmin(axis=1)] * self.coupon[delta.argmin(axis=1)]
        return nkd

    '''
    def coupons_pv(self, rate): 
        ф-ция возвращает массив размера (n, 1) (n - кол-во дат), где элемент явл-ся приведенной стоиомостью купонов на дату 

        * years - разница между купонными датами и рассматриваемыми датами
        * pv - каждый купон приводится к нужной дате
        * pv.sum() - считается сумма приведенных купонов по каждой дате, при этом берется только те купоны, для которых years > 0,
                     т.е. проверяется не был ли выплачен купон до рассматриваемой даты
    '''

    def coupons_pv(self, rate):
        years = (self.coupon_end_date - self.today).astype(int) / 365
        pv = self.coupon / ((1 + rate) ** years)
        return pv.sum(axis=1, where=years > 0).reshape(-1, 1)

    '''
    def face_value_pv(self, rate): 
        ф-ция возвращает массив размера (n, 1) (n - кол-во дат), где элемент явл-ся приведенной стоиомостью номинала на дату 

        * years - разница между датой погашения и рассматриваемых дат
        * pv - приводится поминал к нужной дате
    '''

    def face_value_pv(self, rate):
        years = (self.maturity - self.today).astype(int) / 365
        pv = self.face_value / ((1 + rate) ** years)
        return pv.sum(axis=1, where=years > 0).reshape(-1, 1) / self.nominal.reshape(-1, 1) * 100

    def get_pv(self, rate):
        pv = self.coupons_pv(rate) + self.face_value_pv(rate)
        return pv

    '''
    def get_ytm(self): 
        ф-ция возвращает массив размера (n, 1) (n - кол-во дат), где элемент явл-ся доходностью к погашению/офрете на дату

        Скорее всего вычисления здесь можно ускорить. 
        На данный момент используется цикл для того, чтобы найти корни ф-ии для каждой даты. 
        Думаю, что можно использовать векторные вычисления, чтобы не использовать цикл, но пока не знаю как это реализовать 

    '''

    def get_ytm(self):
        from scipy import optimize
        ytm_arr = []
        nkd = self.nkd()
        for i in range(len(self.today)):
            ytm = optimize.newton(lambda rate: self.get_pv(rate)[i] - self.market_price[i] - nkd[i], x0=0, maxiter=100)
            ytm_arr.append(ytm)
        return np.array(ytm_arr).reshape(-1, 1)

    '''
    def get_duration(self): 
        ф-ция возвращает массив размера (n, 1) (n - кол-во дат), где элемент явл-ся приведенной стоиомостью номинала на дату 

        * years - разница между датой погашения и рассматриваемых дат
        * ytm - считаются дох-ти к погашению/оферте
        * pv - считается числитель формулы дюрации (1ое слагаемое только купоны, 2ое - только номинал)
    '''

    def get_duration(self):
        years = (self.coupon_end_date - self.today).astype(int) / 365
        years_fv = (self.maturity - self.today).astype(int) / 365
        ytm = self.get_ytm()
        pv = ((self.coupon * years / ((1 + ytm) ** years)).sum(axis=1, where=years > 0).reshape(-1,
                                                                                                1) +  # тут тока купоны
              ((self.face_value * years_fv / ((1 + ytm) ** years_fv))).sum(axis=1, where=years_fv > 0).reshape(-1,
                                                                                                               1) / self.nominal.reshape(
                    -1, 1) * 100)  # тут номинал
        dur = pv / self.get_pv(ytm)
        return dur

    '''
    def summa(params, t):
        вспомогательная ф-ция, кот-ая исп-ся при подсчете значения GCurve
        исп-ся декортор @staticmethod для того, чтобы можно было воспользоваться данной функцией без объявления экземпляра класса
    '''

    @staticmethod
    def summa(params, t):
        list_a = [0, 0.6]
        list_b = [0.6]
        k = 1.6
        s = 0
        for i in range(1, 10):
            if i > 1:
                list_b.append(list_b[-1] * k)

                if i > 2:
                    list_a.append(list_a[-1] + list_a[1] * k ** (i - 2))

            s += params[f'G{i}'] * np.exp(-(t - list_a[i - 1]) ** 2 / (list_b[i - 1] ** 2))

        return s

    '''
    def GCurve(params, t):
        подсчет значения GCurve

        * params - параметры функции для выбранных дат (в виде датафрейма)
        * t - дюрация
    '''

    @staticmethod
    def GCurve(params, t):

        s = (params['B1'] + (params['B2'] + params['B3']) * (params['T1'] / t) *
             (1 - np.exp(-t / params['T1'])) - params['B3'] * np.exp(-t / params['T1']) + Bond.summa(params, t))

        return np.exp(s / 10000) - 1

    '''
    def get_model_price(self):
        подсчет модельной цены

        * gcurve_params - импортируется sql таблица с параметрами для GCurve за период с t1 по t2

        * params - выбирается индекс нужной даты из gcurve_params (если для нужной даты параметров нет, 
          то выбирается другая ближ. дата)

        * t - записывается дюрация на каждую дату

        * gytm - подсчитаются значения GCurve на каждую дату

        * gspread - считается спред между ytm и gytm на каждую дату

        * model_ytm - считается модельная ytm как сумма gspread и значения GCurve на каждую дату, 
          при этом для каждой даты исп-ся дюрация на последнюю дату

        * model_price - считается модельная цена с исп-ем модельной ytm 
    '''

    def get_model_price(self):

        gcurve_params = pd.read_sql_query(...)

        params = gcurve_params.iloc[
            np.argmin(abs(np.array(gcurve_params.index, dtype='datetime64[D]') - self.today), axis=1)]
        t = self.get_duration().flatten()
        gytm = Bond.GCurve(params, t)

        gspread = self.get_ytm().flatten() - gytm
        model_ytm = gspread + Bond.GCurve(params, t[0])
        model_price = self.get_pv(model_ytm.values.reshape(-1, 1)) - self.nkd().reshape(-1, 1)
        self.df_prices['model price'] = model_price

        return self.df_prices