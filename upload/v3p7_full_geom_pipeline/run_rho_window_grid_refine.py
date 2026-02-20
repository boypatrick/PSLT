#!/usr/bin/env python3
# Reproduce D=6/12 rho-window check (rho_max=1,2,4) and grid convergence (coarse/mid/fine) for Dijkstra24 and FMM.
import math, heapq, time
import numpy as np
import pandas as pd

import os

def _interp(x, xs, ys):
    return float(np.interp(float(x), xs, ys))

def load_tcoh_curve(curve_csv="tcoh_geom_curve_axis1d.csv"):
    # Load geometry-closed tcoh curve derived from axis-1D scan.
    # Used here only to provide omega1(D) and C_star for computing tcoh_geom
    # from the 2D Smin computed in this script.
    if not os.path.exists(curve_csv):
        return None
    df = pd.read_csv(curve_csv)
    required = {"D","omega1","C_star"}
    if not required.issubset(set(df.columns)):
        return None
    return df.sort_values("D").reset_index(drop=True)

def make_acceptance_table(df_raw, Sz_v3p1_map):
    rows_out=[]
    for algo in ("dijkstra24","fmm"):
        col = "Smin_"+algo
        for D in sorted(df_raw["D"].unique()):
            sub = df_raw[df_raw["D"]==D]
            def pick(level, rho):
                q = sub[(sub["level"]==level) & (np.isclose(sub["rho_max"], rho))]
                if len(q)!=1:
                    raise RuntimeError(f"Missing/duplicate row for D={D}, level={level}, rho_max={rho}")
                return float(q[col].iloc[0]), float(q["Sz_grid"].iloc[0])
            S_f4, Sz_f4 = pick("fine", 4.0)
            S_f1, _     = pick("fine", 1.0)
            S_f2, _     = pick("fine", 2.0)
            S_mid, _    = pick("mid", 4.0)
            S_coarse, _ = pick("coarse", 4.0)

            rel_window_rho1 = (S_f1 - S_f4)/S_f4
            rel_window_rho2 = (S_f2 - S_f4)/S_f4
            rel_mid = (S_mid - S_f4)/S_f4
            rel_coarse = (S_coarse - S_f4)/S_f4
            rel_Szgrid = (Sz_f4 - S_f4)/S_f4

            Sz_v3 = float(Sz_v3p1_map.get(int(D), np.nan))
            rel_Szv3p1 = (Sz_v3 - S_f4)/S_f4 if np.isfinite(Sz_v3) else np.nan

            rows_out.append(dict(
                D=int(D), algo=algo,
                S_fine_rho4=S_f4, S_fine_rho1=S_f1, S_fine_rho2=S_f2,
                rel_window_rho1=rel_window_rho1, rel_window_rho2=rel_window_rho2,
                S_mid_rho4=S_mid, S_coarse_rho4=S_coarse, rel_mid=rel_mid, rel_coarse=rel_coarse,
                Sz_grid_fine_rho4=Sz_f4, rel_Szgrid=rel_Szgrid,
                Sz_v3p1=Sz_v3, rel_Szv3p1=rel_Szv3p1
            ))
    return pd.DataFrame(rows_out).sort_values(["D","algo"]).reset_index(drop=True)

def add_tcoh_columns(df_acc, curve_df=None, C_star_fallback=1.829255651765622):
    # Adds omega1_est, C_star, DeltaE_pred, tcoh_geom and log10_tcoh_geom.
    out=df_acc.copy()
    if curve_df is None:
        omega_map={6:0.7215, 12:0.722418}
        out["omega1_est"]=out["D"].map(omega_map).astype(float)
        out["C_star"]=C_star_fallback
    else:
        Ds=curve_df["D"].values.astype(float)
        Ws=curve_df["omega1"].values.astype(float)
        C_star=float(curve_df["C_star"].iloc[0])
        out["omega1_est"]=out["D"].apply(lambda d: _interp(d, Ds, Ws))
        out["C_star"]=C_star

    out["Smin_used"]=out["S_fine_rho4"]
    out["DeltaE_pred"]=out["C_star"]*np.exp(-out["Smin_used"])
    out["tcoh_geom"]=(2*math.pi/out["C_star"])*out["omega1_est"]*np.exp(out["Smin_used"])
    out["log10_tcoh_geom"]=np.log10(out["tcoh_geom"])
    return out

def omega_rz(rho, z, D, a, eps):
    r1 = np.sqrt(rho*rho + (z - D/2.0)**2 + eps*eps)
    r2 = np.sqrt(rho*rho + (z + D/2.0)**2 + eps*eps)
    return 1.0 + a*(1.0/r1 + 1.0/r2)

def laplace_omega_rz(rho, z, D, a, eps):
    r1sq = rho*rho + (z - D/2.0)**2 + eps*eps
    r2sq = rho*rho + (z + D/2.0)**2 + eps*eps
    lap1 = -3.0*eps*eps/(r1sq**(2.5))
    lap2 = -3.0*eps*eps/(r2sq**(2.5))
    return a*(lap1+lap2)

def U_rz(rho, z, D, a, eps, m0, xi):
    Om = omega_rz(rho, z, D, a, eps)
    lapOm = laplace_omega_rz(rho, z, D, a, eps)
    return m0*m0*(Om*Om - 1.0) + (1.0-6.0*xi)*(lapOm/Om)

def find_component_mask(mask, z):
    Nz = mask.shape[1]
    j0 = int(np.argmin(np.abs(z-0.0)))
    comp = np.zeros_like(mask, dtype=bool)
    left_idx = np.full(mask.shape[0], -1, dtype=int)
    right_idx = np.full(mask.shape[0], -1, dtype=int)
    for i in range(mask.shape[0]):
        row = mask[i]
        if not row[j0]:
            true_idx = np.where(row)[0]
            if len(true_idx)==0:
                continue
            jseed = int(true_idx[np.argmin(np.abs(true_idx-j0))])
        else:
            jseed = j0
        jL=jseed
        while jL-1>=0 and row[jL-1]:
            jL-=1
        jR=jseed
        while jR+1<Nz and row[jR+1]:
            jR+=1
        comp[i, jL:jR+1]=True
        left_idx[i]=jL
        right_idx[i]=jR
    return comp, left_idx, right_idx

neighbor_offsets=[(di,dj) for di in range(-2,3) for dj in range(-2,3) if not (di==0 and dj==0)]

def smin_dijkstra(W, comp_mask, dr, dz, left_idx, right_idx):
    Nr, Nz = W.shape
    INF=1e300
    dist=np.full((Nr,Nz), INF, dtype=np.float64)
    visited=np.zeros((Nr,Nz), dtype=bool)
    heap=[]
    target=np.zeros((Nr,Nz), dtype=bool)
    for i in range(Nr):
        j=left_idx[i]
        if j>=0 and comp_mask[i,j]:
            dist[i,j]=0.0
            heapq.heappush(heap,(0.0,i,j))
        jr=right_idx[i]
        if jr>=0 and comp_mask[i,jr]:
            target[i,jr]=True
    while heap:
        d,i,j=heapq.heappop(heap)
        if visited[i,j]:
            continue
        visited[i,j]=True
        if target[i,j]:
            return d
        Wi=W[i,j]
        for di,dj in neighbor_offsets:
            ni=i+di; nj=j+dj
            if ni<0 or ni>=Nr or nj<0 or nj>=Nz:
                continue
            if not comp_mask[ni,nj] or visited[ni,nj]:
                continue
            ds=math.hypot(di*dr, dj*dz)
            w_edge=0.5*(Wi+W[ni,nj])*ds
            nd=d+w_edge
            if nd<dist[ni,nj]:
                dist[ni,nj]=nd
                heapq.heappush(heap,(nd,ni,nj))
    return math.nan

def smin_fmm(W, comp_mask, dr, dz, left_idx, right_idx):
    Nr, Nz = W.shape
    INF=1e300
    T=np.full((Nr,Nz), INF, dtype=np.float64)
    state=np.zeros((Nr,Nz), dtype=np.int8) # 0 far, 1 trial, 2 accepted
    heap=[]
    target=set()
    for i in range(Nr):
        j=left_idx[i]
        if j>=0 and comp_mask[i,j]:
            T[i,j]=0.0
            heapq.heappush(heap,(0.0,i,j))
            state[i,j]=1
        jr=right_idx[i]
        if jr>=0 and comp_mask[i,jr]:
            target.add((i,jr))
    def update(i,j):
        if not comp_mask[i,j]:
            return
        tx=[]
        if i-1>=0 and state[i-1,j]==2: tx.append(T[i-1,j])
        if i+1<Nr and state[i+1,j]==2: tx.append(T[i+1,j])
        ty=[]
        if j-1>=0 and state[i,j-1]==2: ty.append(T[i,j-1])
        if j+1<Nz and state[i,j+1]==2: ty.append(T[i,j+1])
        if not tx and not ty:
            return
        a=0.0; b=0.0; c=-(W[i,j]**2)
        tmax=-INF
        if tx:
            Tx=min(tx)
            a += 1.0/(dr*dr)
            b += -2.0*Tx/(dr*dr)
            c += Tx*Tx/(dr*dr)
            tmax=max(tmax,Tx)
        if ty:
            Ty=min(ty)
            a += 1.0/(dz*dz)
            b += -2.0*Ty/(dz*dz)
            c += Ty*Ty/(dz*dz)
            tmax=max(tmax,Ty)
        disc=b*b-4*a*c
        if disc<0:
            cand=[]
            if tx: cand.append(Tx + W[i,j]*dr)
            if ty: cand.append(Ty + W[i,j]*dz)
            Tnew=min(cand)
        else:
            Tnew=(-b+math.sqrt(disc))/(2*a)
            if Tnew < tmax:
                cand=[]
                if tx: cand.append(Tx + W[i,j]*dr)
                if ty: cand.append(Ty + W[i,j]*dz)
                Tnew=min(cand)
        if Tnew < T[i,j]:
            T[i,j]=Tnew
            heapq.heappush(heap,(Tnew,i,j))
            state[i,j]=1

    while heap:
        t,i,j=heapq.heappop(heap)
        if state[i,j]==2:
            continue
        if t!=T[i,j]:
            continue
        state[i,j]=2
        if (i,j) in target:
            return t
        for ni,nj in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
            if 0<=ni<Nr and 0<=nj<Nz and comp_mask[ni,nj] and state[ni,nj]!=2:
                update(ni,nj)
    return math.nan

def compute_case(D, E, z1, z2, dr, dz, rho_max, PAD, params):
    a,eps,m0,xi=params
    rho=np.arange(0, rho_max+1e-12, dr)
    z=np.arange(z1-PAD, z2+PAD+1e-12, dz)
    R,Z=np.meshgrid(rho,z,indexing='ij')
    U=U_rz(R,Z,D,a,eps,m0,xi)
    mask=U>E
    comp,left_idx,right_idx=find_component_mask(mask,z)
    W=np.sqrt(np.maximum(U-E,0.0))
    Sd=smin_dijkstra(W, comp, dr, dz, left_idx, right_idx)
    Sf=smin_fmm(W, comp, dr, dz, left_idx, right_idx)
    comp0=comp[0]
    Sz=float(np.trapz(np.sqrt(np.maximum(U[0]-E,0.0))*comp0.astype(float), z)) if np.any(comp0) else math.nan
    return Sd,Sf,Sz

if __name__=='__main__':
    PAD=1.0
    params=(0.040,0.10,1.0,0.14)
    # These E,z1,z2 should be taken from your v3p1 output for n=1 (physical_gap):
    meta={
      6: {'E': -0.4794374275180444, 'z1': -2.8592217504924333, 'z2':  2.8592217504924333},
      12:{'E': -0.4781115780400631, 'z1': -5.858427411179166,  'z2':  5.858427411179166}
    }
    grid=[('coarse',0.04,0.02),('mid',0.03,0.015),('fine',0.02,0.01)]
    rho_list=[1.0,2.0,4.0]
    rows=[]
    for D in (6,12):
        E=meta[D]['E']; z1=meta[D]['z1']; z2=meta[D]['z2']
        for lvl,dr,dz in grid:
            for rho_max in rho_list:
                if lvl!='fine' and rho_max!=4.0:
                    continue
                Sd,Sf,Sz=compute_case(D,E,z1,z2,dr,dz,rho_max,PAD,params)
                rows.append(dict(D=D,level=lvl,dr=dr,dz=dz,rho_max=rho_max,PAD=PAD,
                                 Smin_dijkstra24=Sd,Smin_fmm=Sf,Sz_grid=Sz))
    df_raw = pd.DataFrame(rows)
    df_raw.to_csv('cases_raw_results_repro.csv',index=False)
    print('Wrote cases_raw_results_repro.csv')

    # Acceptance table (same schema as final_acceptance_table.csv).
    # Sz_v3p1 values are taken from the v3p1 axis scan output (n=1 physical_gap):
    Sz_v3p1_map = {6: 4.281585, 12: 8.507335}
    df_acc = make_acceptance_table(df_raw, Sz_v3p1_map)
    df_acc.to_csv('final_acceptance_table_repro.csv', index=False)
    print('Wrote final_acceptance_table_repro.csv')

    # Geometry-closed coherence time (t_coh) columns.
    curve_df = load_tcoh_curve('tcoh_geom_curve_axis1d.csv')
    df_acc_t = add_tcoh_columns(df_acc, curve_df=curve_df)
    df_acc_t.to_csv('final_acceptance_table_with_tcoh_repro.csv', index=False)
    print('Wrote final_acceptance_table_with_tcoh_repro.csv')

