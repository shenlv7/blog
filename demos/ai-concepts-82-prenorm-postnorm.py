#!/usr/bin/env python3
"""Pre-LN vs Post-LN：残差流范数增长、逐层梯度量级、单层删除敏感度。
纯 numpy、无训练，全部是在随机初始化网络上的真实前向/反向测量。
"""
import numpy as np

rng = np.random.default_rng(20261114)
D, HID, B = 256, 512, 64
EPS = 1e-5


def gelu(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def dgelu(x):
    c = np.sqrt(2.0 / np.pi)
    t = np.tanh(c * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * c * (1 + 3 * 0.044715 * x ** 2)


def ln_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + EPS)
    yhat = (x - mu) * inv
    return yhat * g + b, yhat, inv


def ln_bwd(dy, yhat, inv, g):
    dyh = dy * g
    dx = inv * (dyh - dyh.mean(-1, keepdims=True)
                - yhat * (dyh * yhat).mean(-1, keepdims=True))
    return dx, (dy * yhat).sum(0), dy.sum(0)


def init_block():
    lim1 = np.sqrt(6.0 / (D + HID))
    lim2 = np.sqrt(6.0 / (HID + D))
    return dict(W1=rng.uniform(-lim1, lim1, (D, HID)), b1=np.zeros(HID),
                W2=rng.uniform(-lim2, lim2, (HID, D)), b2=np.zeros(D),
                g=np.ones(D), be=np.zeros(D))


def make_net(L):
    return [init_block() for _ in range(L)]


def forward(net, x, kind, drop=None):
    """drop=l：第 l 层只留恒等路径（Pre-LN 就是直接跳过，Post-LN 只剩一层 LN）。"""
    caches, norms = [], [float(np.linalg.norm(x, axis=1).mean())]
    for l, p in enumerate(net):
        if drop == l:
            if kind == 'post':
                x, yhat, inv = ln_fwd(x, p['g'], p['be'])
            caches.append(None)
            norms.append(float(np.linalg.norm(x, axis=1).mean()))
            continue
        if kind == 'pre':
            y, yhat, inv = ln_fwd(x, p['g'], p['be'])
            a = y @ p['W1'] + p['b1']
            h = gelu(a)
            z = h @ p['W2'] + p['b2']
            caches.append(dict(kind='pre', xin=x, lin=y, a=a, h=h, yhat=yhat, inv=inv, p=p))
            x = x + z
        else:
            a = x @ p['W1'] + p['b1']
            h = gelu(a)
            z = h @ p['W2'] + p['b2']
            u = x + z
            out, yhat, inv = ln_fwd(u, p['g'], p['be'])
            caches.append(dict(kind='post', xin=x, lin=x, a=a, h=h, yhat=yhat, inv=inv, p=p))
            x = out
        norms.append(float(np.linalg.norm(x, axis=1).mean()))
    return x, caches, norms


def backward(caches, r):
    """loss = mean(out · r)。返回每层 ||dW||（含 dW1、dW2）。"""
    dout = np.repeat((r / B)[None, :], B, axis=0)   # loss = sum_b(out_b·r)/B
    out = []
    for c in reversed(caches):
        if c is None:
            continue
        p = c['p']
        if c['kind'] == 'pre':
            dz = dout
            dW2 = c['h'].T @ dz
            dh = dz @ p['W2'].T
            da = dh * dgelu(c['a'])
            dW1 = c['lin'].T @ da
            dy = da @ p['W1'].T
            dx_ln, _, _ = ln_bwd(dy, c['yhat'], c['inv'], p['g'])
            dout = dout + dx_ln
        else:
            du, _, _ = ln_bwd(dout, c['yhat'], c['inv'], p['g'])
            dW2 = c['h'].T @ du
            dh = du @ p['W2'].T
            da = dh * dgelu(c['a'])
            dW1 = c['lin'].T @ da
            dout = du + da @ p['W1'].T   # 残差路径 + 穿过 MLP 的那条路径
        out.append(dict(n=float(np.sqrt((dW1 ** 2).sum() + (dW2 ** 2).sum())), dW1=dW1, dW2=dW2))
    return out[::-1]


x0 = rng.standard_normal((B, D))
r = rng.standard_normal(D)
L = 24
print(f'=== 随机初始化 · {L} 层 · d={D} hid={HID} batch={B} loss=mean(out·r) ===\n')

res = {}
for kind, name in (('pre', 'Pre-LN '), ('post', 'Post-LN')):
    net = make_net(L)
    out, caches, norms = forward(net, x0, kind)
    g = backward(caches, r)
    base = np.linalg.norm(out)
    abl = []
    for l in range(L):
        o2, _, _ = forward(net, x0, kind, drop=l)
        abl.append(float(np.linalg.norm(o2 - out) / base))
    print(f'--- {name} ---')
    for l in range(L):
        print(f'  L{l+1:02d} 流范数 {norms[l+1]:7.3f} (对上一步 {(norms[l+1]/norms[l]-1)*100:+6.2f}%)'
              f'  ||dW|| {g[l]["n"]:.3e}  单层删除 Δ {abl[l]*100:6.3f}%')
    print(f'  首层/末层梯度 {g[0]["n"]/g[-1]["n"]:.2e}   层间最大/最小梯度 {max(x["n"] for x in g)/min(x["n"] for x in g):.2e}')
    print(f'  流范数 {norms[0]:.2f} -> {norms[-1]:.2f} ({norms[-1]/norms[0]:.2f}x)'
          f'   删除敏感度 最大 {max(abl)*100:.3f}% / 中位 {np.median(abl)*100:.3f}%\n')
    res[kind] = (norms, g, abl)

print('=== 深度扫描：首层/末层梯度比 ===')
for L2 in (6, 12, 24, 48):
    row = []
    for kind in ('pre', 'post'):
        net = make_net(L2)
        out, caches, _ = forward(net, x0, kind)
        g = backward(caches, r)
        row.append(f'{kind}: {g[0]["n"]/g[-1]["n"]:.2e}')
    print(f'  L={L2:2d}  ' + '   '.join(row))

# --- 自检：LN 反向 + 手动反传是否和数值梯度一致（小尺寸，float64） ---
def grad_check(kind):
    global D, HID, B
    D0, H0, B0 = D, HID, B
    D, HID, B = 8, 16, 4
    net = make_net(2)
    x = rng.standard_normal((B, D)); rr = rng.standard_normal(D)

    def loss_of():
        o, _, _ = forward(net, x, kind)
        return float((o @ rr).sum() / B)

    _, caches, _ = forward(net, x, kind)
    g = backward(caches, rr)
    worst = 0.0
    for li in (0, 1):
        for key in ('W1', 'W2'):
            W = net[li][key]
            idx = [tuple(rng.integers(0, s) for s in W.shape) for _ in range(6)]
            num = []
            for ij in idx:
                old = W[ij]; eps = 1e-6
                W[ij] = old + eps; lp = loss_of()
                W[ij] = old - eps; lm = loss_of()
                W[ij] = old
                num.append((lp - lm) / (2 * eps))
            # 该层的解析梯度
            dWnum = np.array(num)
            dbg = g[li]['dW1'] if key == 'W1' else g[li]['dW2']
            ana = np.array([dbg[ij] for ij in idx])
            rel = np.abs(ana - dWnum) / (np.abs(dWnum) + 1e-12)
            worst = max(worst, float(rel.max()))
    print(f'  梯度自检 {kind}: 最大相对误差 {worst:.2e}')
    D, HID, B = D0, H0, B0


print('\n=== 反传正确性自检（有限差分对照） ===')
grad_check('pre')
grad_check('post')
