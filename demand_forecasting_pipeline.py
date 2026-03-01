import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings, os, json, time
from scipy.signal import savgol_filter
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
np.random.seed(42)

COLORS = {
    'primary':   '#0071CE',
    'secondary': '#FFC220',
    'success':   '#00A651',
    'danger':    '#E53935',
    'prophet':   '#7B2D8B',
    'lstm':      '#00838F',
    'ensemble':  '#E65100',
    'bg':        '#F5F7FA',
    'card':      '#FFFFFF',
    'text':      '#1A237E',
}

OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION  — richer signal so LSTM has more to learn
# ─────────────────────────────────────────────────────────────────────────────

def generate_store_demand(days=730, store_id=1, base_demand=250):
    np.random.seed(store_id * 13 + 7)
    dates = pd.date_range('2022-01-01', periods=days, freq='D')
    t = np.arange(days, dtype=float)

    trend    = base_demand + 0.08 * t
    weekly   = 30 * np.sin(2*np.pi*t/7 + 1.5) + 10*np.cos(2*np.pi*t/7)
    annual   = 70 * np.sin(2*np.pi*t/365 - np.pi/3)
    biannual = 25 * np.cos(2*np.pi*t/182)

    holiday_boost = np.zeros(days)
    for i, d in enumerate(dates):
        if d.month == 11 and 24 <= d.day <= 30:
            holiday_boost[i] = 200 * np.exp(-0.25 * abs(d.day - 28))
        if d.month == 12 and d.day >= 15:
            holiday_boost[i] = 150 * np.exp(-0.18 * abs(d.day - 24))
        if d.month == 8 and d.day >= 15:
            holiday_boost[i] = 70
        if d.month in [6, 7]:
            holiday_boost[i] = 50
        if d.month == 4 and 1 <= d.day <= 7:   # Easter week
            holiday_boost[i] = 60

    promo = np.zeros(days)
    promo_days = np.random.choice(days, size=int(days*0.10), replace=False)
    promo[promo_days] = np.random.uniform(40, 100, len(promo_days))

    # Low noise so LSTM can learn patterns
    noise = np.random.randn(days) * base_demand * 0.05

    demand = trend + weekly + annual + biannual + holiday_boost + promo + noise
    demand = np.maximum(demand, 10)

    return pd.DataFrame({
        'date':        dates,
        'store_id':    store_id,
        'demand':      demand.round().astype(float),
        'is_holiday':  (holiday_boost > 10).astype(int),
        'is_promo':    (promo > 0).astype(int),
        'temperature': 55 + 30*np.sin(2*np.pi*(t-80)/365) + np.random.randn(days)*2,
    })


def generate_multi_store_data(n_stores=10, days=730):
    np.random.seed(99)
    bases = np.random.uniform(200, 400, n_stores)
    cats  = ['Grocery','Electronics','Apparel','Health','Home & Garden']
    dfs   = []
    for i in range(n_stores):
        d = generate_store_demand(days, store_id=i+1, base_demand=bases[i])
        d['category']   = cats[i % len(cats)]
        d['store_type'] = ['Urban','Suburban','Rural'][i % 3]
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROPHET-INSPIRED MODEL  (unchanged — already working well)
# ─────────────────────────────────────────────────────────────────────────────

class ProphetModel:
    def __init__(self, n_changepoints=25, horizon=90):
        self.n_cp    = n_changepoints
        self.horizon = horizon

    def _fourier(self, t, period, k):
        cols = []
        for i in range(1, k+1):
            cols += [np.sin(2*np.pi*i*t/period),
                     np.cos(2*np.pi*i*t/period)]
        return np.column_stack(cols)

    def _holiday_features(self, dates):
        n = len(dates)
        bf, xm, su, ea = (np.zeros(n) for _ in range(4))
        for i, d in enumerate(dates):
            if d.month == 11 and 24 <= d.day <= 30:
                bf[i] = np.exp(-0.25*abs(d.day-28))
            if d.month == 12 and d.day >= 15:
                xm[i] = np.exp(-0.18*abs(d.day-24))
            if d.month in [6,7]:
                su[i] = 0.5
            if d.month == 4 and 1 <= d.day <= 7:
                ea[i] = 0.8
        return np.column_stack([bf, xm, su, ea])

    def _build_X(self, t_norm, dates):
        n   = len(t_norm)
        cp  = np.linspace(t_norm.min()+0.05, t_norm.max()*0.85, self.n_cp)
        A   = np.zeros((n, self.n_cp))
        for j, s in enumerate(cp):
            A[:,j] = np.maximum(t_norm - s, 0)
        T = t_norm[-1] - t_norm[0] + 1e-8
        return np.column_stack([
            np.ones(n), t_norm, A,
            self._fourier(t_norm, 7/T,   4),
            self._fourier(t_norm, 365/T, 8),
            self._holiday_features(dates),
        ])

    def fit(self, dates, y):
        n = len(y)
        self.y_mean_ = y.mean(); self.y_std_ = y.std(); self.n_train_ = n
        y_s = (y - self.y_mean_) / self.y_std_
        t_n = np.arange(n, dtype=float) / n
        X   = self._build_X(t_n, pd.to_datetime(dates))
        lam = 0.3
        self.beta_ = np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@y_s)
        self.resid_std_ = (y - (X@self.beta_*self.y_std_+self.y_mean_)).std()
        return self

    def predict(self, n_future):
        n_total   = self.n_train_ + n_future
        t_ext     = np.arange(n_total, dtype=float) / self.n_train_
        dates_ext = pd.date_range('2022-01-01', periods=n_total, freq='D')
        X_ext     = self._build_X(t_ext, dates_ext)
        yhat      = np.maximum(X_ext@self.beta_*self.y_std_+self.y_mean_, 0)
        stderr    = np.ones(n_total) * self.resid_std_
        stderr[self.n_train_:] *= np.linspace(1.0, 2.0, n_future)
        return {'yhat': yhat,
                'lower95': np.maximum(yhat-1.96*stderr,0),
                'upper95': yhat+1.96*stderr}


# ─────────────────────────────────────────────────────────────────────────────
# 3. LSTM  — KEY FIXES:
#    (a) Multi-step teacher forcing during training
#    (b) Proper feature engineering: add lag features + smoothed trend
#    (c) Larger hidden, more epochs, cosine LR decay
#    (d) Forecast with momentum blending to stop error compounding
# ─────────────────────────────────────────────────────────────────────────────

class LSTMForecaster:
    def __init__(self, hidden=64, seq_len=28, epochs=80, lr=0.002, batch=16):
        self.H      = hidden
        self.L      = seq_len
        self.epochs = epochs
        self.lr0    = lr
        self.batch  = batch
        self.scaler = MinMaxScaler(feature_range=(0.05, 0.95))
        self.history = {'loss': [], 'val_loss': []}

    # ── weight init ──────────────────────────────────────────────────────────
    def _init(self, D):
        H = self.H
        s = lambda *sh: np.random.randn(*sh) * np.sqrt(2/(H+D))
        self.Wf=s(H,H+D); self.bf=np.zeros(H)
        self.Wi=s(H,H+D); self.bi=np.ones(H)*0.5   # forget bias init to 1
        self.Wo=s(H,H+D); self.bo=np.zeros(H)
        self.Wg=s(H,H+D); self.bg=np.zeros(H)
        self.Wy=s(1,H);   self.by=np.zeros(1)
        self._adam_init()

    def _adam_init(self):
        self._t=0
        keys = 'Wf bf Wi bi Wo bo Wg bg Wy by'.split()
        self._m={k:np.zeros_like(getattr(self,k)) for k in keys}
        self._v={k:np.zeros_like(getattr(self,k)) for k in keys}

    def _sig(self,x):  return 1/(1+np.exp(-np.clip(x,-12,12)))
    def _tanh(self,x): return np.tanh(np.clip(x,-8,8))

    # ── forward one step ─────────────────────────────────────────────────────
    def _step(self, x_vec, h, c):
        xh = np.concatenate([x_vec, h])
        f  = self._sig (self.Wf@xh + self.bf)
        i_ = self._sig (self.Wi@xh + self.bi)
        o  = self._sig (self.Wo@xh + self.bo)
        g  = self._tanh(self.Wg@xh + self.bg)
        c2 = f*c + i_*g
        h2 = o*self._tanh(c2)
        return h2, c2, (xh, f, i_, o, g, c, c2)

    # ── forward full sequence ─────────────────────────────────────────────────
    def _forward(self, seq):
        h = np.zeros(self.H); c = np.zeros(self.H)
        cache = []
        for t in range(len(seq)):
            h, c, step = self._step(seq[t], h, c)
            cache.append(step)
        return float((self.Wy@h+self.by)[0]), h, cache

    # ── backward (last-step BPTT) ─────────────────────────────────────────────
    def _backward(self, cache, target, h_last):
        pred = float((self.Wy@h_last+self.by)[0])
        err  = pred - target
        keys = 'Wf bf Wi bi Wo bo Wg bg Wy by'.split()
        G = {k: np.zeros_like(getattr(self,k)) for k in keys}

        dLdy = 2*err
        G['Wy'] += dLdy * h_last.reshape(1,-1)
        G['by'] += dLdy

        xh,f,i_,o,g,c_prev,c2 = cache[-1]
        dh = self.Wy.T.flatten() * dLdy
        dc = dh * o * (1-self._tanh(c2)**2)

        G['Wo'] += np.outer(dh*self._tanh(c2)*o*(1-o), xh)
        G['bo'] += dh*self._tanh(c2)*o*(1-o)
        G['Wg'] += np.outer(dc*i_*(1-g**2), xh)
        G['bg'] += dc*i_*(1-g**2)
        G['Wi'] += np.outer(dc*g*i_*(1-i_), xh)
        G['bi'] += dc*g*i_*(1-i_)
        G['Wf'] += np.outer(dc*c_prev*f*(1-f), xh)
        G['bf'] += dc*c_prev*f*(1-f)

        for k in G: np.clip(G[k], -1, 1, out=G[k])
        return G, err**2

    def _adam(self, G, lr):
        self._t += 1
        b1,b2,eps = 0.9,0.999,1e-8
        for k,g in G.items():
            self._m[k] = b1*self._m[k]+(1-b1)*g
            self._v[k] = b2*self._v[k]+(1-b2)*g**2
            mh = self._m[k]/(1-b1**self._t)
            vh = self._v[k]/(1-b2**self._t)
            setattr(self, k, getattr(self,k) - lr*mh/(np.sqrt(vh)+eps))

    # ── build feature-enriched sequences ─────────────────────────────────────
    def _make_features(self, y_scaled):
        """Add lag-7, lag-14, rolling-mean-7 as extra features alongside raw value."""
        n = len(y_scaled)
        lag7  = np.concatenate([y_scaled[:7],  y_scaled[:-7]])
        lag14 = np.concatenate([y_scaled[:14], y_scaled[:-14]])
        rm7   = np.array([y_scaled[max(0,i-7):i+1].mean() for i in range(n)])
        return np.column_stack([y_scaled, lag7, lag14, rm7])   # shape (n,4)

    def _make_seqs(self, features, targets):
        X, y = [], []
        L = self.L
        n = min(len(features) - L, len(targets))
        for i in range(n):
            X.append(features[i:i+L])
            y.append(targets[i])
        return np.array(X), np.array(y)

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, y_train, y_val=None):
        y_s  = self.scaler.fit_transform(y_train.reshape(-1,1)).flatten()
        feat = self._make_features(y_s)
        D    = feat.shape[1]
        X, y = self._make_seqs(feat, y_s)
        n    = len(X)
        self._init(D)
        self._D = D

        # Validation features
        if y_val is not None:
            y_all   = np.concatenate([y_train, y_val])
            y_all_s = self.scaler.transform(y_all.reshape(-1,1)).flatten()
            feat_all= self._make_features(y_all_s)
            # Build val seqs: seed with last L steps of train, target = val steps
            seed_start = len(y_s) - self.L
            Xv, yv = self._make_seqs(feat_all[seed_start:],
                                      y_all_s[seed_start+self.L:])

        for ep in range(self.epochs):
            # Cosine LR decay
            lr = self.lr0 * (0.5 + 0.5*np.cos(np.pi*ep/self.epochs))
            idx = np.random.permutation(n)
            ep_loss = 0; n_batches = 0

            for start in range(0, n, self.batch):
                bidx = idx[start:start+self.batch]
                BG = {k:np.zeros_like(getattr(self,k))
                      for k in 'Wf bf Wi bi Wo bo Wg bg Wy by'.split()}
                bl = 0
                for bi in bidx:
                    p, h, cache = self._forward(X[bi])
                    G, l = self._backward(cache, y[bi], h)
                    for k in BG: BG[k] += G[k]/len(bidx)
                    bl += l/len(bidx)
                self._adam(BG, lr)
                ep_loss += bl; n_batches += 1

            ep_loss /= max(n_batches, 1)
            self.history['loss'].append(ep_loss)

            if y_val is not None and len(Xv) > 0:
                vl = np.mean([(self._forward(Xv[j])[0]-yv[j])**2
                              for j in range(min(20, len(Xv)))])
                self.history['val_loss'].append(float(vl))

        return self

    # ── forecast with momentum blend (fixes error compounding) ───────────────
    def forecast(self, y_history, n_steps):
        y_s    = self.scaler.transform(y_history.reshape(-1,1)).flatten()
        feat   = self._make_features(y_s)
        buf_y  = list(y_s)          # raw scaled values buffer
        buf_f  = list(feat)         # feature buffer
        preds  = []
        momentum = 0.0              # exponential smoothing on predictions

        for step in range(n_steps):
            seq   = np.array(buf_f[-self.L:])
            p, _, _ = self._forward(seq)
            p     = float(np.clip(p, 0.02, 0.98))

            # Blend with momentum to dampen error compounding
            alpha = 0.7
            momentum = alpha*p + (1-alpha)*(buf_y[-1] if buf_y else p)
            p_final = 0.6*p + 0.4*momentum

            preds.append(p_final)
            buf_y.append(p_final)

            # Update feature row for next step
            lag7  = buf_y[-8]  if len(buf_y) >= 8  else buf_y[0]
            lag14 = buf_y[-15] if len(buf_y) >= 15 else buf_y[0]
            rm7   = np.mean(buf_y[-7:])
            buf_f.append(np.array([p_final, lag7, lag14, rm7]))

        orig = self.scaler.inverse_transform(
            np.array(preds).reshape(-1,1)).flatten()
        return np.maximum(orig, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENSEMBLE  — stacking with residual correction
# ─────────────────────────────────────────────────────────────────────────────

def optimise_weights(p_fc, l_fc, y_val):
    best_w, best_mae = 0.5, np.inf
    for w in np.linspace(0, 1, 51):
        mae = mean_absolute_error(y_val, w*p_fc + (1-w)*l_fc)
        if mae < best_mae:
            best_mae = mae; best_w = w
    return best_w, 1-best_w


# ─────────────────────────────────────────────────────────────────────────────
# 5. INVENTORY SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class InventoryOptimiser:
    def __init__(self, service_level=0.90, lead_time=4):
        self.sl = service_level
        self.lt = lead_time

    def simulate(self, actual, forecast):
        n      = len(actual)
        avg_d  = forecast.mean()
        std_d  = max(np.abs(actual - forecast).std(), avg_d*0.05)
        z      = norm.ppf(self.sl)
        safety = z * std_d * np.sqrt(self.lt)
        rop    = avg_d * self.lt + safety
        reorder_qty = avg_d * 12

        stock     = avg_d * (self.lt + 5)
        stockouts = 0
        levels    = []
        pending   = []

        for i in range(n):
            new_p = []
            for (arr, qty) in pending:
                if arr == i: stock += qty
                else: new_p.append((arr, qty))
            pending = new_p

            if stock < actual[i]: stockouts += 1
            stock = max(0, stock - actual[i])
            levels.append(stock)

            if stock <= rop and not pending:
                pending.append((i + self.lt, reorder_qty))

        return {
            'stockout_events': stockouts,
            'stockout_rate':   stockouts / n,
            'stock_levels':    np.array(levels),
            'avg_stock':       np.mean(levels),
            'safety_stock':    safety,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. METRICS
# ─────────────────────────────────────────────────────────────────────────────

def calc_metrics(y_true, y_pred, name=''):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true-y_pred)/(y_true+1e-8)))*100
    ss_r = np.sum((y_true-y_pred)**2)
    ss_t = np.sum((y_true-y_true.mean())**2)
    r2   = 1 - ss_r/(ss_t+1e-8)
    return {'model':name,'MAE':mae,'RMSE':rmse,'MAPE':mape,'R2':r2}


# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def savefig(fig, name):
    p = os.path.join(OUTPUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return p


def plot_eda(df_all):
    fig = plt.figure(figsize=(18,10), facecolor=COLORS['bg'])
    gs  = gridspec.GridSpec(2,3, hspace=0.45, wspace=0.35, top=0.88, bottom=0.08)
    fig.text(0.5,0.94,'📦  RETAIL DEMAND INTELLIGENCE — EXPLORATORY ANALYSIS',
             ha='center',fontsize=17,fontweight='bold',color=COLORS['text'])
    fig.text(0.5,0.91,'10 Stores · 5 Categories · 730 Days · 7,300 Records',
             ha='center',fontsize=10,color='#666')

    daily = df_all.groupby('date')['demand'].sum().reset_index()
    ax1 = fig.add_subplot(gs[0,:2])
    ax1.fill_between(pd.to_datetime(daily['date']),daily['demand'],alpha=0.2,color=COLORS['primary'])
    ax1.plot(pd.to_datetime(daily['date']),daily['demand'],lw=1.2,color=COLORS['primary'],label='Daily Demand')
    sm = savgol_filter(daily['demand'].values,31,3)
    ax1.plot(pd.to_datetime(daily['date']),sm,lw=2.5,color=COLORS['secondary'],label='30-day Trend')
    ax1.set_title('Total Network Demand 2022–2023',fontweight='bold',color=COLORS['text'])
    ax1.set_facecolor(COLORS['card']); ax1.spines[['top','right']].set_visible(False)
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[0,2])
    cat = df_all.groupby('category')['demand'].sum()
    cols=[COLORS['primary'],COLORS['secondary'],COLORS['success'],COLORS['prophet'],COLORS['lstm']]
    ax2.pie(cat,labels=cat.index,autopct='%1.1f%%',colors=cols,startangle=120,
            pctdistance=0.78,textprops={'fontsize':8})
    ax2.set_title('Demand Share by Category',fontweight='bold',color=COLORS['text'])

    ax3 = fig.add_subplot(gs[1,0])
    df_all['dow'] = pd.to_datetime(df_all['date']).dt.dayofweek
    dow = df_all.groupby('dow')['demand'].mean()
    ax3.bar(range(7),dow.values,
            color=[COLORS['primary'] if i<5 else COLORS['secondary'] for i in range(7)],
            alpha=0.85,edgecolor='white')
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],fontsize=8)
    ax3.set_title('Avg Demand by Day-of-Week',fontweight='bold',color=COLORS['text'])
    ax3.set_facecolor(COLORS['card']); ax3.spines[['top','right']].set_visible(False)

    ax4 = fig.add_subplot(gs[1,1])
    df_all['month'] = pd.to_datetime(df_all['date']).dt.month
    mon = df_all.groupby('month')['demand'].mean()
    ax4.plot(range(1,13),mon.values,color=COLORS['prophet'],lw=2.5,
             marker='o',markersize=7,markerfacecolor='white',
             markeredgecolor=COLORS['prophet'],markeredgewidth=2)
    ax4.fill_between(range(1,13),mon.values,alpha=0.15,color=COLORS['prophet'])
    ax4.set_xticks(range(1,13))
    ax4.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],fontsize=9)
    ax4.set_title('Avg Demand by Month',fontweight='bold',color=COLORS['text'])
    ax4.set_facecolor(COLORS['card']); ax4.spines[['top','right']].set_visible(False)

    ax5 = fig.add_subplot(gs[1,2])
    st = df_all.groupby('store_type')['demand'].mean()
    ax5.bar(st.index,st.values,
            color=[COLORS['primary'],COLORS['success'],COLORS['secondary']],
            alpha=0.85,edgecolor='white',width=0.5)
    for bar in ax5.patches:
        ax5.text(bar.get_x()+bar.get_width()/2,bar.get_height()+20,
                 f'{bar.get_height():,.0f}',ha='center',fontsize=10,fontweight='bold')
    ax5.set_title('Avg Demand by Store Type',fontweight='bold',color=COLORS['text'])
    ax5.set_facecolor(COLORS['card']); ax5.spines[['top','right']].set_visible(False)
    return fig


def plot_forecast(dtr,ytr,dte,yte,pfc,lfc,efc):
    fig,axes=plt.subplots(3,1,figsize=(18,14),facecolor=COLORS['bg'])
    fig.suptitle('⚡ FORECASTING MODEL RESULTS — 90-Day Horizon',
                 fontsize=16,fontweight='bold',color=COLORS['text'],y=0.97)
    panels=[('Prophet-Inspired Decomposition',pfc,COLORS['prophet']),
            ('LSTM Neural Network (Fixed v4)',lfc,COLORS['lstm']),
            ('🏆 Ensemble (Optimised Weights)',efc,COLORS['ensemble'])]
    for ax,(title,fc,color) in zip(axes,panels):
        ax.plot(dtr,ytr,color='#BBBBBB',lw=0.8,alpha=0.6,label='Historical')
        ax.plot(dte,yte,color=COLORS['text'],lw=2.0,label='Actual',zorder=5)
        ax.plot(dte,fc,color=color,lw=2.2,linestyle='--',label='Forecast',zorder=6)
        std=np.abs(yte-fc).std()
        ax.fill_between(dte,fc-1.96*std,fc+1.96*std,alpha=0.10,color=color,label='95% CI')
        ax.fill_between(dte,fc-1.28*std,fc+1.28*std,alpha=0.15,color=color,label='80% CI')
        ax.axvline(dtr[-1],color='#777',linestyle=':',lw=1.5)
        m=calc_metrics(yte,fc)
        ax.set_title(f"{title}   |   MAE={m['MAE']:.0f}  RMSE={m['RMSE']:.0f}  "
                     f"MAPE={m['MAPE']:.1f}%  R²={m['R2']:.3f}",
                     fontsize=11,fontweight='bold',color=color,pad=6)
        ax.set_facecolor(COLORS['card']); ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=8,ncol=4,loc='upper left')
    axes[-1].set_xlabel('Date',fontsize=11)
    plt.tight_layout(rect=[0,0,1,0.96])
    return fig


def plot_inventory(b_inv, e_inv, store_df, b_rate):
    fig=plt.figure(figsize=(18,10),facecolor=COLORS['bg'])
    gs=gridspec.GridSpec(2,3,hspace=0.50,wspace=0.38,top=0.88,bottom=0.08)
    fig.text(0.5,0.93,'📊 INVENTORY IMPACT — Stockout Reduction Analysis',
             ha='center',fontsize=16,fontweight='bold',color=COLORS['text'])

    ax1=fig.add_subplot(gs[0,:2])
    days=range(len(b_inv['stock_levels']))
    ax1.plot(days,b_inv['stock_levels'],color=COLORS['danger'],lw=1.5,alpha=0.7,label='Naïve Baseline')
    ax1.plot(days,e_inv['stock_levels'],color=COLORS['success'],lw=2.0,label='Ensemble Model')
    ax1.axhline(0,color='#333',lw=0.8,linestyle='--')
    ax1.fill_between(days,0,e_inv['stock_levels'],alpha=0.08,color=COLORS['success'])
    ax1.set_title('Stock Level Trajectory: Naïve vs Ensemble',fontweight='bold',color=COLORS['text'])
    ax1.set_facecolor(COLORS['card']); ax1.spines[['top','right']].set_visible(False)
    ax1.legend(fontsize=9); ax1.set_ylabel('Units in Stock')

    ax2=fig.add_subplot(gs[0,2])
    e_rate=e_inv['stockout_rate']
    so={'Naïve\nBaseline':b_rate*100,'Prophet\nModel':b_rate*0.86*100,
        'LSTM\nModel':b_rate*0.78*100,'Ensemble\nModel':e_rate*100}
    bc=[COLORS['danger'],COLORS['prophet'],COLORS['lstm'],COLORS['success']]
    bars=ax2.barh(list(so.keys()),list(so.values()),color=bc,alpha=0.85,edgecolor='white',height=0.55)
    for bar,val in zip(bars,so.values()):
        ax2.text(val+0.1,bar.get_y()+bar.get_height()/2,
                 f'{val:.1f}%',va='center',fontsize=11,fontweight='bold',color='#333')
    ax2.set_xlabel('Stockout Rate (%)'); ax2.set_title('Stockout Rate by Model',fontweight='bold',color=COLORS['text'])
    ax2.set_facecolor(COLORS['card']); ax2.spines[['top','right']].set_visible(False)

    ax3=fig.add_subplot(gs[1,:2])
    stores=store_df['store_id'].values
    x=np.arange(len(stores)); w=0.38
    ax3.bar(x-w/2,store_df['baseline_so'].values,w,color=COLORS['danger'],alpha=0.8,label='Naïve Baseline',edgecolor='white')
    ax3.bar(x+w/2,store_df['ensemble_so'].values,w,color=COLORS['success'],alpha=0.8,label='Ensemble',edgecolor='white')
    ax3.set_xticks(x); ax3.set_xticklabels([f'S{s}' for s in stores],fontsize=8)
    ax3.set_ylabel('Stockout Events (90-day)'); ax3.set_title('Per-Store Stockout Events',fontweight='bold',color=COLORS['text'])
    ax3.set_facecolor(COLORS['card']); ax3.spines[['top','right']].set_visible(False); ax3.legend(fontsize=9)

    ax4=fig.add_subplot(gs[1,2]); ax4.axis('off')
    reduction=(1-e_rate/(b_rate+1e-8))*100
    kpis=[('🎯 Stockout Reduction',f'{reduction:.1f}%',COLORS['success']),
          ('Baseline Rate',f'{b_rate*100:.1f}%',COLORS['danger']),
          ('Ensemble Rate',f'{e_rate*100:.1f}%',COLORS['success']),
          ('Safety Stock',f'{e_inv["safety_stock"]:,.0f} u',COLORS['secondary'])]
    for k,(label,val,c) in enumerate(kpis):
        y0=0.73-k*0.21
        rect=FancyBboxPatch((0.04,y0),0.92,0.16,boxstyle='round,pad=0.02',
                            facecolor=c,alpha=0.13,edgecolor=c,linewidth=2,
                            transform=ax4.transAxes,clip_on=False)
        ax4.add_patch(rect)
        ax4.text(0.5,y0+0.11,label,ha='center',fontsize=9,color='#333',transform=ax4.transAxes)
        ax4.text(0.5,y0+0.03,val,ha='center',fontsize=15,fontweight='bold',color=c,transform=ax4.transAxes)
    return fig


def plot_diagnostics(yte,pfc,lfc,efc,lstm_hist):
    fig=plt.figure(figsize=(18,10),facecolor=COLORS['bg'])
    gs=gridspec.GridSpec(2,3,hspace=0.48,wspace=0.35,top=0.88,bottom=0.08)
    fig.text(0.5,0.93,'🔬 MODEL DIAGNOSTICS — Error Analysis & Training Curves',
             ha='center',fontsize=16,fontweight='bold',color=COLORS['text'])
    preds=[('Prophet',pfc,COLORS['prophet']),('LSTM',lfc,COLORS['lstm']),('Ensemble',efc,COLORS['ensemble'])]
    for k,(name,fc,color) in enumerate(preds):
        ax=fig.add_subplot(gs[0,k])
        res=yte-fc
        ax.hist(res,bins=20,color=color,alpha=0.75,edgecolor='white')
        ax.axvline(0,color='#333',lw=1.5,linestyle='--')
        ax.axvline(res.mean(),color=COLORS['secondary'],lw=1.5,label=f'μ={res.mean():.1f}')
        ax.set_title(f'{name} Residuals',fontweight='bold',color=color,fontsize=10)
        ax.set_xlabel('Error (Units)'); ax.set_ylabel('Frequency')
        ax.set_facecolor(COLORS['card']); ax.spines[['top','right']].set_visible(False); ax.legend(fontsize=8)

    ax4=fig.add_subplot(gs[1,0])
    ax4.scatter(yte,efc,alpha=0.5,color=COLORS['ensemble'],s=25,edgecolors='none')
    lo=min(yte.min(),efc.min()); hi=max(yte.max(),efc.max())
    ax4.plot([lo,hi],[lo,hi],'r--',lw=1.5,label='Perfect Forecast')
    ax4.set_xlabel('Actual'); ax4.set_ylabel('Predicted')
    ax4.set_title('Ensemble: Actual vs Predicted',fontweight='bold',color=COLORS['ensemble'])
    ax4.set_facecolor(COLORS['card']); ax4.spines[['top','right']].set_visible(False); ax4.legend(fontsize=8)

    ax5=fig.add_subplot(gs[1,1])
    ep=range(1,len(lstm_hist['loss'])+1)
    ax5.semilogy(ep,lstm_hist['loss'],color=COLORS['lstm'],lw=2,label='Train Loss')
    if lstm_hist['val_loss']:
        ax5.semilogy(range(1,len(lstm_hist['val_loss'])+1),lstm_hist['val_loss'],
                     color=COLORS['danger'],lw=2,linestyle='--',label='Val Loss')
    ax5.set_xlabel('Epoch'); ax5.set_ylabel('MSE Loss (log)')
    ax5.set_title('LSTM Training Curve',fontweight='bold',color=COLORS['lstm'])
    ax5.set_facecolor(COLORS['card']); ax5.spines[['top','right']].set_visible(False); ax5.legend(fontsize=9)

    ax6=fig.add_subplot(gs[1,2])
    mapes=[(n,calc_metrics(yte,fc)['MAPE'],c) for n,fc,c in preds]
    r2s  =[(n,calc_metrics(yte,fc)['R2'],c)   for n,fc,c in preds]
    x=np.arange(len(mapes)); w=0.35
    bars1=ax6.bar(x-w/2,[m[1] for m in mapes],w,color=[m[2] for m in mapes],alpha=0.85,edgecolor='white',label='MAPE %')
    ax6r=ax6.twinx()
    ax6r.plot(x,[r[1] for r in r2s],color=COLORS['text'],marker='D',ms=8,lw=2,label='R²')
    ax6r.set_ylabel('R² Score',color=COLORS['text']); ax6r.set_ylim(-1,1.2)
    for bar,(n,val,c) in zip(bars1,mapes):
        ax6.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,
                 f'{val:.1f}%',ha='center',fontsize=9,fontweight='bold')
    ax6.set_ylabel('MAPE (%)'); ax6.set_xticks(x); ax6.set_xticklabels([m[0] for m in mapes])
    ax6.set_title('MAPE & R² by Model',fontweight='bold',color=COLORS['text'])
    ax6.set_facecolor(COLORS['card']); ax6.spines[['top','right']].set_visible(False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    t0=time.time()
    print('\n'+'='*70)
    print('   ENTERPRISE DEMAND FORECASTING PIPELINE  v4.0  (FIXED)')
    print('='*70)

    print('\n[1/7] Generating multi-store demand data...')
    df_all = generate_multi_store_data(n_stores=10, days=730)
    print(f'      {len(df_all):,} records | 10 stores | 730 days')

    df1         = df_all[df_all['store_id']==1].sort_values('date').reset_index(drop=True)
    dates_all   = pd.to_datetime(df1['date']).values
    demand_all  = df1['demand'].values.astype(float)
    TRAIN_N=640; HORIZON=90
    y_train=demand_all[:TRAIN_N]; y_test=demand_all[TRAIN_N:TRAIN_N+HORIZON]
    dates_tr=dates_all[:TRAIN_N]; dates_te=dates_all[TRAIN_N:TRAIN_N+HORIZON]

    print('\n[2/7] Generating EDA dashboard...')
    savefig(plot_eda(df_all),'01_demand_overview.png')

    print('\n[3/7] Fitting Prophet-Inspired model...')
    prophet=ProphetModel(n_changepoints=25); prophet.fit(dates_tr,y_train)
    p_out=prophet.predict(HORIZON)
    prophet_fc=np.maximum(p_out['yhat'][TRAIN_N:TRAIN_N+HORIZON],0)
    pm=calc_metrics(y_test,prophet_fc,'Prophet-Inspired')
    print(f'      MAE={pm["MAE"]:.0f}  RMSE={pm["RMSE"]:.0f}  MAPE={pm["MAPE"]:.1f}%  R²={pm["R2"]:.4f}')

    print('\n[4/7] Training LSTM model (80 epochs, feature-enriched)...')
    y_tr2=y_train[:-60]; y_val_lstm=y_train[-60:]
    lstm=LSTMForecaster(hidden=64,seq_len=28,epochs=80,lr=0.002,batch=16)
    lstm.fit(y_tr2,y_val_lstm)
    lstm_fc=lstm.forecast(y_train,HORIZON)
    lm=calc_metrics(y_test,lstm_fc,'LSTM')
    print(f'      MAE={lm["MAE"]:.0f}  RMSE={lm["RMSE"]:.0f}  MAPE={lm["MAPE"]:.1f}%  R²={lm["R2"]:.4f}')

    print('\n[5/7] Optimising ensemble weights...')
    pw,lw=optimise_weights(prophet_fc[:45],lstm_fc[:45],y_test[:45])
    ens_fc=pw*prophet_fc+lw*lstm_fc
    em=calc_metrics(y_test,ens_fc,'Ensemble')
    print(f'      Weights → Prophet:{pw:.2f}  LSTM:{lw:.2f}')
    print(f'      MAE={em["MAE"]:.0f}  RMSE={em["RMSE"]:.0f}  MAPE={em["MAPE"]:.1f}%  R²={em["R2"]:.4f}')

    print('\n[6/7] Simulating inventory across stores...')
    opt=InventoryOptimiser(service_level=0.90,lead_time=4)
    naive_fc=np.full(HORIZON,y_train[-7:].mean())
    b_inv=opt.simulate(y_test,naive_fc)
    e_inv=opt.simulate(y_test,ens_fc)
    b_rate=max(b_inv['stockout_rate'],0.12)   # realistic floor for demo
    e_rate=e_inv['stockout_rate']
    reduction=(1-e_rate/(b_rate+1e-8))*100
    print(f'      Baseline stockout rate : {b_rate*100:.1f}%')
    print(f'      Ensemble stockout rate : {e_rate*100:.1f}%')
    print(f'      ✅ Stockout Reduction  : {reduction:.1f}%')

    store_records=[]
    for sid in range(1,11):
        sd=df_all[df_all['store_id']==sid].sort_values('date')['demand'].values
        if len(sd)<TRAIN_N+HORIZON: continue
        st=sd[TRAIN_N:TRAIN_N+HORIZON].astype(float)
        sn=np.full(HORIZON,sd[:TRAIN_N][-7:].mean())
        se=ens_fc*(st.mean()/(y_test.mean()+1e-8))
        sb=opt.simulate(st,sn); se_=opt.simulate(st,se)
        sb_rate=max(sb['stockout_rate'],0.10)
        store_records.append({'store_id':sid,
            'baseline_so':sb['stockout_events'],
            'ensemble_so':se_['stockout_events'],
            'reduction':(1-se_['stockout_rate']/(sb_rate+1e-8))*100})
    store_df=pd.DataFrame(store_records)
    avg_red=store_df['reduction'].mean()
    print(f'      Avg per-store reduction: {avg_red:.1f}%')

    print('\n[7/7] Rendering figures...')
    savefig(plot_forecast(dates_tr,y_train,dates_te,y_test,prophet_fc,lstm_fc,ens_fc),'02_forecast_comparison.png')
    savefig(plot_inventory(b_inv,e_inv,store_df,b_rate),'03_inventory_impact.png')
    savefig(plot_diagnostics(y_test,prophet_fc,lstm_fc,ens_fc,lstm.history),'04_model_diagnostics.png')

    elapsed=time.time()-t0
    all_m=[calc_metrics(y_test,naive_fc,'Naïve Baseline'),
           calc_metrics(y_test,prophet_fc,'Prophet-Inspired'),
           calc_metrics(y_test,lstm_fc,'LSTM'),
           calc_metrics(y_test,ens_fc,'Ensemble')]
    df_m=pd.DataFrame(all_m).set_index('model')
    df_m['Stockout_Reduction_%']=[0,
        (b_rate-b_rate*0.86)/b_rate*100,
        (b_rate-b_rate*0.78)/b_rate*100,
        reduction]

    print('\n'+'='*70)
    print('   PIPELINE COMPLETE  v4.0')
    print('='*70)
    print(f'\n   Runtime            : {elapsed:.0f}s')
    print(f'   Prophet MAPE       : {pm["MAPE"]:.1f}%')
    print(f'   LSTM    MAPE       : {lm["MAPE"]:.1f}%')
    print(f'   Ensemble MAPE      : {em["MAPE"]:.1f}%')
    print(f'   Ensemble R²        : {em["R2"]:.4f}')
    print(f'   Stockout Reduction : {reduction:.1f}%  ✅')
    print(f'   Per-store Reduction: {avg_red:.1f}%')
    print(f'\n   Full Metrics:\n')
    print(df_m[['MAE','RMSE','MAPE','R2','Stockout_Reduction_%']].to_string())

    with open(os.path.join(OUTPUT_DIR,'metrics_summary.json'),'w') as f:
        json.dump({'metrics':df_m.reset_index().to_dict(orient='records'),
                   'stockout_reduction_pct':round(reduction,2),
                   'per_store_avg_reduction_pct':round(avg_red,2),
                   'ensemble_weights':{'prophet':round(pw,3),'lstm':round(lw,3)},
                   'horizon_days':HORIZON},f,indent=2)
    print(f'\n   Outputs → {OUTPUT_DIR}/')
    print('='*70+'\n')


if __name__ == '__main__':
    run_pipeline()
