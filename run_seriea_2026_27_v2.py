import io
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def norm(x):
    x = '' if pd.isna(x) else str(x)
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii')
    x = x.lower().replace("'", ' ')
    x = re.sub(r'[^a-z0-9 ]+', ' ', x)
    return re.sub(r'\s+', ' ', x).strip()


def key_name(x):
    p = norm(x).split()
    if len(p) > 1 and len(p[-1]) <= 2:
        p = p[:-1]
    return ' '.join(p)


def get(url):
    r = requests.get(url, timeout=45, headers={'User-Agent': 'Mozilla/5.0'})
    r.raise_for_status()
    return r.text


def flat(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(str(v) for v in c if str(v) != 'nan').strip() for c in df.columns]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df


def pick(df, tokens):
    for token in tokens:
        nt = norm(token)
        for c in df.columns:
            cc = norm(c)
            if cc == nt or nt in cc.split():
                return c
    for token in tokens:
        nt = norm(token)
        for c in df.columns:
            if nt in norm(c):
                return c
    return None


def stats_table(html):
    for t in pd.read_html(io.StringIO(html), decimal=',', thousands='.'):
        t = flat(t)
        cols = ' '.join(norm(c) for c in t.columns)
        if ' pv ' in f' {cols} ' and ' mv ' in f' {cols} ' and ' fm ' in f' {cols} ':
            return t
    raise RuntimeError('Fantacalcio statistics table not found')


def clean_stats(t):
    t = flat(t.copy())
    team = pick(t, ['sq', 'squadra', 'team'])
    pv = pick(t, ['pv'])
    mv = pick(t, ['mv'])
    fm = pick(t, ['fm'])
    gf = pick(t, ['gol', 'gf'])
    gs = pick(t, ['gs'])
    rig = pick(t, ['rig'])
    rp = pick(t, ['rp'])
    ass = pick(t, ['ass'])
    amm = pick(t, ['amm'])
    esp = pick(t, ['esp'])
    if team is None or pv is None:
        raise RuntimeError(f'Unexpected table columns: {list(t.columns)}')

    # On Fantacalcio, the Calciatore heading spans decorative columns; the actual
    # player-name column is the last column immediately before Sq.
    team_pos = list(t.columns).index(team)
    before_team = list(t.columns)[:team_pos]
    if not before_team:
        raise RuntimeError('No player-name column before Sq')
    name = before_team[-1]

    def num(c):
        if c is None:
            return pd.Series(0.0, index=t.index)
        s = t[c].astype(str).str.replace(',', '.', regex=False)
        s = s.str.extract(r'(-?\d+(?:\.\d+)?)', expand=False)
        return pd.to_numeric(s, errors='coerce').fillna(0.0)

    out = pd.DataFrame(index=t.index)
    out['Name'] = t[name].astype(str).str.strip()
    out['Team'] = t[team].astype(str).str.strip()
    out['Pv'] = num(pv)
    out['Mv'] = num(mv)
    out['Fm'] = num(fm)
    out['Gf'] = num(gf)
    out['Gs'] = num(gs)
    out['Rp'] = num(rp)
    out['Rc'] = num(rp)
    out['Ass'] = num(ass)
    out['Amm'] = num(amm)
    out['Esp'] = num(esp)
    if rig is not None:
        txt = t[rig].astype(str).str.replace(',', '.', regex=False)
        out['Rp'] = pd.to_numeric(txt.str.extract(r'^\s*(\d+)', expand=False), errors='coerce').fillna(0)
    out['R+'] = 0
    out['R-'] = 0
    out['Au'] = 0
    out = out[out['Name'].str.lower().ne('nan') & out['Name'].str.strip().ne('')]
    return out


def gazzetta_roles(html):
    soup = BeautifulSoup(html, 'html.parser')
    rows = []
    for tr in soup.find_all('tr'):
        cells = [c.get_text(' ', strip=True) for c in tr.find_all(['td', 'th'])]
        idx = next((i for i, v in enumerate(cells) if v.upper() in {'P','D','C','A'}), None)
        if idx is not None and idx >= 2:
            rows.append((cells[idx-1], cells[idx], cells[idx-2]))
    return pd.DataFrame(rows, columns=['NameG','Role','TeamG']).drop_duplicates()


def score(r):
    return (0.5*r.Gf + 0.2*r.Ass - 0.05*r.Amm - 0.1*r.Esp + 0.2*r.Mv +
            0.2*r.Rp - 0.5*r.Gs + 0.5*r.Rc + 0.5*r.Pv)


def main():
    cur = clean_stats(stats_table(get('https://www.fantacalcio.it/statistiche-serie-a/2026-27')))
    prev = clean_stats(stats_table(get('https://www.fantacalcio.it/statistiche-serie-a/2025-26')))
    roles = gazzetta_roles(get('https://www.gazzetta.it/calcio/fantanews/lista-giocatori-fantacalcio-serie-a-2026-27/'))

    print('Current sample names:', cur.Name.head(8).tolist())
    print('Gazzetta role sample:', roles[['NameG','Role']].head(8).values.tolist())

    cur['key'] = cur.Name.map(key_name)
    prev['key'] = prev.Name.map(key_name)
    roles['key'] = roles.NameG.map(key_name)
    cur = cur.sort_values('Pv', ascending=False).drop_duplicates('key')
    prev = prev.sort_values('Pv', ascending=False).drop_duplicates('key')

    rg = roles.groupby('key')['Role'].agg(lambda x: x.iloc[0] if x.nunique() == 1 else '')
    cur['R'] = cur['key'].map(rg).fillna('')

    # Returning-player fallback to the repository's previous role map.
    if Path('Fantacalcio_stat.csv').exists():
        old = pd.read_csv('Fantacalcio_stat.csv')
        old['key'] = old.Name.map(key_name)
        om = old.groupby('key')['R'].agg(lambda x: x.iloc[0] if x.nunique() == 1 else '')
        mask = ~cur.R.isin(['P','D','C','A'])
        cur.loc[mask, 'R'] = cur.loc[mask, 'key'].map(om).fillna('')

    valid = cur.R.isin(['P','D','C','A']).sum()
    print('Current players with valid Classic roles:', int(valid), 'of', len(cur))
    if valid < 100:
        bad = cur.loc[~cur.R.isin(['P','D','C','A']), 'Name'].head(30).tolist()
        raise RuntimeError(f'Role match insufficient; unmatched sample={bad}')

    prev = prev.set_index('key')
    rows = []
    counts = ['Pv','Gf','Gs','Rp','Rc','R+','R-','Ass','Amm','Esp','Au']
    for _, r in cur.iterrows():
        z = r.copy()
        if r['key'] in prev.index:
            p = prev.loc[r['key']]
            a, b = float(r.Pv), float(p.Pv)
            d = a + b
            if d:
                z['Mv'] = (float(r.Mv)*a + float(p.Mv)*b)/d
                z['Fm'] = (float(r.Fm)*a + float(p.Fm)*b)/d
            for c in counts:
                z[c] = float(r[c]) + float(p[c])
        rows.append(z)
    d = pd.DataFrame(rows)
    d = d[d.R.isin(['P','D','C','A']) & (d.Pv > 0)].copy()
    d['score_internal'] = d.apply(score, axis=1)

    caps = {'P':30,'D':80,'C':80,'A':60}
    parts = []
    for role, cap in caps.items():
        g = d[d.R == role].sort_values('score_internal', ascending=False).head(cap)
        if len(g) < min(cap, 20):
            raise RuntimeError(f'Too few {role}: {len(g)}')
        parts.append(g)
    d = pd.concat(parts, ignore_index=True)
    d = d.sort_values(['R','score_internal'], ascending=[True,False]).reset_index(drop=True)
    d.insert(0, 'Id', np.arange(1, len(d)+1))
    d['Rm'] = d.R.map({'P':'Por','D':'Dc','C':'Cc','A':'Pc'})
    cols = ['Id','R','Rm','Name','Team','Pv','Mv','Fm','Gf','Gs','Rp','Rc','R+','R-','Ass','Amm','Esp','Au']
    d[cols].to_csv('Fantacalcio_stat.csv', index=False)
    d[cols].to_csv('Fantacalcio_stat_2026_27_combined.csv', index=False)
    print('Prepared:', len(d), d.R.value_counts().to_dict())
    print('Method: current 2026/27 roster; counts summed with 2025/26; MV/FM appearance-weighted.')


if __name__ == '__main__':
    main()
