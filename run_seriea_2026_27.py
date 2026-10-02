import io
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def _norm(s):
    s = '' if pd.isna(s) else str(s)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = s.lower().replace("'", ' ')
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def _surname_key(s):
    s = _norm(s)
    parts = s.split()
    if len(parts) > 1 and len(parts[-1]) <= 2:
        parts = parts[:-1]
    return ' '.join(parts)


def fetch_html(url):
    r = requests.get(url, timeout=45, headers={
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/130 Safari/537.36'
    })
    r.raise_for_status()
    return r.text


def flatten_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(str(x) for x in tup if str(x) != 'nan').strip() for tup in df.columns]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df


def find_stats_table(html):
    tables = pd.read_html(io.StringIO(html), decimal=',', thousands='.')
    for t in tables:
        t = flatten_cols(t)
        cols = ' | '.join(t.columns).lower()
        if 'pv' in cols and 'mv' in cols and 'fm' in cols and ('gol' in cols or 'gf' in cols):
            return t
    raise RuntimeError('Could not find Fantacalcio statistics table')


def colpick(df, tokens, default=None):
    for token in tokens:
        nt = _norm(token)
        for c in df.columns:
            cc = _norm(c)
            if cc == nt or nt in cc.split():
                return c
    for token in tokens:
        nt = _norm(token)
        for c in df.columns:
            if nt in _norm(c):
                return c
    return default


def clean_stats(df):
    df = flatten_cols(df.copy())
    name_c = colpick(df, ['calciatore', 'name'])
    role_c = colpick(df, ['r', 'ruolo', 'role'])
    team_c = colpick(df, ['sq', 'squadra', 'team'])
    pv_c = colpick(df, ['pv'])
    mv_c = colpick(df, ['mv'])
    fm_c = colpick(df, ['fm'])
    gol_c = colpick(df, ['gol', 'gf'])
    gs_c = colpick(df, ['gs'])
    rig_c = colpick(df, ['rig'])
    rp_c = colpick(df, ['rp'])
    ass_c = colpick(df, ['ass'])
    amm_c = colpick(df, ['amm'])
    esp_c = colpick(df, ['esp'])
    if name_c is None or pv_c is None:
        raise RuntimeError(f'Unexpected Fantacalcio columns: {list(df.columns)}')

    out = pd.DataFrame(index=df.index)
    out['Name'] = df[name_c].astype(str).str.strip()
    out['R'] = df[role_c].astype(str).str.strip().str.upper() if role_c else ''
    out['Team'] = df[team_c].astype(str).str.strip() if team_c else ''

    def num(c):
        if c is None:
            return pd.Series([0.0] * len(df), index=df.index)
        s = df[c].astype(str).str.replace(',', '.', regex=False)
        s = s.str.extract(r'(-?\d+(?:\.\d+)?)', expand=False)
        return pd.to_numeric(s, errors='coerce').fillna(0.0)

    for target, c in [('Pv', pv_c), ('Mv', mv_c), ('Fm', fm_c), ('Gf', gol_c),
                      ('Gs', gs_c), ('Rp', rp_c), ('Ass', ass_c), ('Amm', amm_c), ('Esp', esp_c)]:
        out[target] = num(c)

    if rig_c is not None:
        rigtxt = df[rig_c].astype(str).str.replace(',', '.', regex=False)
        out['Rp'] = pd.to_numeric(rigtxt.str.extract(r'^\s*(\d+)', expand=False), errors='coerce').fillna(out['Rp'])
    out['Rc'] = num(rp_c)
    out['R+'] = 0
    out['R-'] = 0
    out['Au'] = 0
    out = out[out['Name'].str.lower().ne('nan') & out['Name'].str.strip().ne('')]
    return out


def find_gazzetta_roles(html):
    tables = pd.read_html(io.StringIO(html))
    for t in tables:
        t = flatten_cols(t)
        name_c = next((c for c in t.columns if 'giocatore' in _norm(c)), None)
        role_c = next((c for c in t.columns if 'ruolo' in _norm(c)), None)
        team_c = next((c for c in t.columns if _norm(c) in {'sqd', 'squadra'} or 'sqd' in _norm(c)), None)
        if name_c and role_c:
            out = pd.DataFrame({'NameG': t[name_c].astype(str).str.strip(),
                                'Role': t[role_c].astype(str).str.strip().str.upper(),
                                'TeamG': t[team_c].astype(str).str.strip() if team_c else ''})
            out = out[out['Role'].isin(['P', 'D', 'C', 'A'])]
            if not out.empty:
                return out
    return pd.DataFrame(columns=['NameG', 'Role', 'TeamG'])


def score_row(r):
    return (0.5*r['Gf'] + 0.2*r['Ass'] - 0.05*r['Amm'] - 0.1*r['Esp'] +
            0.2*r['Mv'] + 0.2*r['Rp'] - 0.5*r['Gs'] + 0.5*r['Rc'] + 0.5*r['Pv'])


def main():
    current_url = 'https://www.fantacalcio.it/statistiche-serie-a/2026-27'
    previous_url = 'https://www.fantacalcio.it/statistiche-serie-a/2025-26'
    gazzetta_url = 'https://www.gazzetta.it/calcio/fantanews/lista-giocatori-fantacalcio-serie-a-2026-27/'

    cur = clean_stats(find_stats_table(fetch_html(current_url)))
    prev = clean_stats(find_stats_table(fetch_html(previous_url)))
    cur['key'] = cur['Name'].map(_surname_key)
    prev['key'] = prev['Name'].map(_surname_key)
    cur = cur.sort_values('Pv', ascending=False).drop_duplicates('key')
    prev = prev.sort_values('Pv', ascending=False).drop_duplicates('key')

    # Fantacalcio's table normally carries the Classic role. Fill any missing roles from Gazzetta.
    cur.loc[~cur['R'].isin(['P', 'D', 'C', 'A']), 'R'] = ''
    if cur['R'].eq('').any():
        try:
            roles = find_gazzetta_roles(fetch_html(gazzetta_url))
        except Exception:
            roles = pd.DataFrame(columns=['NameG', 'Role', 'TeamG'])
        if not roles.empty:
            roles['key'] = roles['NameG'].map(_surname_key)
            role_groups = roles.groupby('key')['Role'].agg(lambda x: x.iloc[0] if x.nunique() == 1 else '')
            mask = cur['R'].eq('')
            cur.loc[mask, 'R'] = cur.loc[mask, 'key'].map(role_groups).fillna('')

    # Final fallback for returning players: repository historical role.
    repo_csv = Path('Fantacalcio_stat.csv')
    if repo_csv.exists() and cur['R'].eq('').any():
        old = pd.read_csv(repo_csv)
        old['key'] = old['Name'].map(_surname_key)
        old_role = old.groupby('key')['R'].agg(lambda x: x.iloc[0] if x.nunique() == 1 else '')
        mask = cur['R'].eq('')
        cur.loc[mask, 'R'] = cur.loc[mask, 'key'].map(old_role).fillna('')

    valid_current_roles = int(cur['R'].isin(['P', 'D', 'C', 'A']).sum())
    print('Current table columns parsed successfully; valid Classic roles:', valid_current_roles)
    if valid_current_roles < 100:
        raise RuntimeError('Too few current players have a valid Classic role; refusing to use stale data.')

    prev2 = prev.set_index('key')
    rows = []
    count_cols = ['Pv', 'Gf', 'Gs', 'Rp', 'Rc', 'R+', 'R-', 'Ass', 'Amm', 'Esp', 'Au']
    for _, r in cur.iterrows():
        p = prev2.loc[r['key']] if r['key'] in prev2.index else None
        out = r.copy()
        if p is not None:
            pv_c, pv_p = float(r['Pv']), float(p['Pv'])
            denom = pv_c + pv_p
            out['Mv'] = ((float(r['Mv'])*pv_c + float(p['Mv'])*pv_p) / denom) if denom else float(r['Mv'])
            out['Fm'] = ((float(r['Fm'])*pv_c + float(p['Fm'])*pv_p) / denom) if denom else float(r['Fm'])
            for c in count_cols:
                out[c] = float(r[c]) + float(p[c])
        rows.append(out)

    combined = pd.DataFrame(rows)
    combined = combined[combined['R'].isin(['P', 'D', 'C', 'A']) & (combined['Pv'] > 0)].copy()
    combined['score_internal'] = combined.apply(score_row, axis=1)

    role_caps = {'P': 30, 'D': 80, 'C': 80, 'A': 60}
    parts = []
    for role, cap in role_caps.items():
        g = combined[combined['R'] == role].sort_values('score_internal', ascending=False).head(cap)
        if not g.empty:
            parts.append(g)
    if len(parts) != 4:
        raise RuntimeError(f'Incomplete role pool: {[p.R.iloc[0] for p in parts]}')
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(['R', 'score_internal'], ascending=[True, False]).reset_index(drop=True)
    combined.insert(0, 'Id', np.arange(1, len(combined)+1))
    combined['Rm'] = combined['R'].map({'P': 'Por', 'D': 'Dc', 'C': 'Cc', 'A': 'Pc'})

    cols = ['Id', 'R', 'Rm', 'Name', 'Team', 'Pv', 'Mv', 'Fm', 'Gf', 'Gs', 'Rp', 'Rc', 'R+', 'R-', 'Ass', 'Amm', 'Esp', 'Au']
    for c in cols:
        if c not in combined:
            combined[c] = 0
    combined[cols].to_csv('Fantacalcio_stat.csv', index=False)
    combined[cols].to_csv('Fantacalcio_stat_2026_27_combined.csv', index=False)

    print(f'Prepared {len(combined)} current Serie A players with 2026/27 + 2025/26 statistics.')
    print('Role counts:', combined['R'].value_counts().to_dict())
    print('Data method: current roster only; counting stats summed across both seasons; MV/FM appearance-weighted.')


if __name__ == '__main__':
    main()
