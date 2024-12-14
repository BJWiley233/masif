import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pyemma

def write_pdb(inds, what, chain):
    norm = np.max(what[inds])
    i_atom = 1
    i_resid = 1
    for i, dind in enumerate(inds):
        fpdb.write('{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n'.format(
            'ATOM',i_atom,
            'GG','','GG',
            chain,i_resid,'',
            xs[dind],ys[dind],zs[dind],
            -1.0*np.log10(what[dind]/norm),what[dind]/norm,
            'K',''))
        i_atom += 1
        if i_atom > 999:
            i_atom = 1
            i_resid += 1
            #eigv[dind,0],mm.pi[dind],
    fpdb.write('TER\n')

if len(sys.argv) != 4:
    raise ValueError('USAGE: python ./msm_gas_grid.py NAME r_min r_max')
name=sys.argv[1]
lags=np.array([1,2,5,10,20,50,75,100,125])
best_lag = 100
dt = 0.1 # [ns] or 1oo ps delle traiettorie fit.dcd (~10.000  steps a traiettoria --> ~1us)
nits = 5 # number of implied timescales
box_min = -20
box_max = +20
dbox = 1.0
rmin_site = float(sys.argv[2])
rmax_site = float(sys.argv[3])
suffix='{0:s}_{1:d}_{2:d}'.format(name,int(rmin_site),int(rmax_site))

pdf = PdfPages('figures_msm_{}.pdf'.format(suffix))
fpdb = open('msm_{}.pdb'.format(suffix), 'wt')

with open('./data_{}.pk'.format(name), 'rb') as fin:
    all_trajs = pickle.load(fin)
trajs = []
max_len = 0
min_len = np.inf
max_x, max_y, max_z = -np.inf, -np.inf, -np.inf
min_x, min_y, min_z =  np.inf,  np.inf,  np.inf
print('Number of trajs in {}: {}'.format(suffix, len(all_trajs)), flush=True)
total_time = 0
for i_traj, traj in enumerate(all_trajs):
    if traj.shape[0] > 100:
        trajs.append(traj)
        #print('\tlen traj {}: {}'.format(i_traj, traj.shape[0]))
        max_x = max(max_x, np.max(traj[:,0]))
        max_y = max(max_y, np.max(traj[:,1]))
        max_z = max(max_z, np.max(traj[:,2]))
        min_x = min(min_x, np.min(traj[:,0]))
        min_y = min(min_y, np.min(traj[:,1]))
        min_z = min(min_z, np.min(traj[:,2]))
        total_time += dt*traj.shape[0]
    max_len = max(max_len, len(traj))
    min_len = min(min_len, len(traj))
max_x = np.ceil(max_x)
max_y = np.ceil(max_y)
max_z = np.ceil(max_z)
min_x = np.floor(min_x)
min_y = np.floor(min_y)
min_z = np.floor(min_z)
print('Number of trajs in {}: {}'.format(suffix, len(trajs)))
print('Longest trajectory in {}: {}'.format(suffix, max_len))
print('Shortest trajectory in {}: {}'.format(suffix, min_len))
print('Total trajectory time in {}: {}'.format(suffix, total_time))
print('x> {}:{}'.format(min_x, max_x))
print('y> {}:{}'.format(min_y, max_y))
print('z> {}:{}'.format(min_z, max_z))

x_edges = np.arange(box_min, box_max+0.5*dbox, dbox)
y_edges = np.arange(box_min, box_max+0.5*dbox, dbox)
z_edges = np.arange(box_min, box_max+0.5*dbox, dbox)
nx = len(z_edges)+1
ny = len(y_edges)+1
nz = len(z_edges)+1
print('Edges: ',x_edges)

dtrajs = []
for i_traj, traj in enumerate(trajs):
    ix = np.digitize(traj[:,0], x_edges)
    iy = np.digitize(traj[:,1], y_edges)
    iz = np.digitize(traj[:,2], z_edges)
    inds = ix + iy*nx + iz*nx*ny
    dtrajs.append(inds.astype(int))
    if i_traj and i_traj % 10000 == 0:
        print('Read {}/{}'.format(i_traj,len(trajs)))
    r = np.sqrt(traj[:,0]**2.0 + traj[:,1]**2.0 + traj[:,2]**2.0)
    print('i_traj = {}'.format(i_traj))
    if np.min(r) < rmin_site:
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        ax.plot(dt*np.arange(len(r)), r, '-')
        ax.plot(dt*np.arange(len(r)), rmin_site*np.ones(len(r)), ':r')
        ax.plot(dt*np.arange(len(r)), rmax_site*np.ones(len(r)), ':r')
        plt.title('{}'.format(i_traj))
        plt.xlabel('Time [ns]')
        plt.ylabel('Radius [A]')
        pdf.savefig()
        plt.close()
print('Done reading')
split_inds = np.cumsum([len(dtraj) for dtraj in dtrajs])[:-1]
dtrajs_array = np.concatenate(dtrajs)
print('Done concatenatig')
vals, dtrajs_mapped = np.unique(dtrajs_array, return_inverse = True)
print('Done mapping')
dtrajs = np.split(dtrajs_mapped, split_inds)
xs, ys, zs, rs = [], [], [], []
x_edges = np.append(x_edges, x_edges[-1]+dbox)
y_edges = np.append(y_edges, y_edges[-1]+dbox)
z_edges = np.append(z_edges, z_edges[-1]+dbox)
for val in vals:
    iz = val // (nx*ny)
    iy = (val - iz*nx*ny) // nx
    ix  = val - iz*nx*ny - iy*nx
    xs.append(x_edges[ix] - 0.5*dbox)
    ys.append(y_edges[iy] - 0.5*dbox)
    zs.append(z_edges[iz] - 0.5*dbox)
    rs.append(np.sqrt(xs[-1]**2+ys[-1]**2+zs[-1]**2))
print('Done splitting')
print('Number of discretized trajs: {}'.format(len(dtrajs)))
xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)
rs = np.array(rs)
print('Radius: {} - {}'.format(np.min(rs), np.max(rs)))

f = plt.figure()
ax = f.add_subplot(1,1,1)
h, e = np.histogram(rs, bins = 100)
b = 0.5 * (e[1:] + e[:-1])
h = h/np.sum(h)
h = h/np.power(b, 2)
ax.plot(b, -np.log(h), 'b-')
pdf.savefig()
plt.xlabel('r [A]')
plt.ylabel('-np.log(P)')
plt.close()

print('Computing MSMs...', flush = True)
taus = np.nan*np.ones((len(lags), nits))
for i_lag, lag in enumerate(lags):
    mm = pyemma.msm.estimate_markov_model(dtrajs, lag, sparse = True)
    taus[i_lag,:] = mm.timescales(nits) # they are already normalized by lag
    print(lag, taus[i_lag,:])

f = plt.figure()
ax = f.add_subplot(1,1,1)
for iits in range(nits):
    ax.plot(dt*lags, dt*taus[:,iits], 'o-')
plt.xlabel('lag [ns]')
plt.ylabel('tau [ns]')
plt.title(suffix)
pdf.savefig()
plt.yscale('log')
pdf.savefig()
plt.close()

print('Computing final MSM...', flush = True)
mm = pyemma.msm.estimate_markov_model(dtrajs, best_lag, sparse = True)
taus = mm.timescales(nits)
print('Selected model', best_lag, taus, flush = True)

f = plt.figure()
ax = f.add_subplot(1,1,1)
ax.plot(np.arange(nits), dt*taus, 'o-')
plt.xlabel('#eigv.')
plt.ylabel('tau [ns]')
plt.title(suffix)
pdf.savefig()
plt.close()

write_pdb(range(len(xs)), rs, 'A')

xs = xs[mm.active_set]
ys = ys[mm.active_set]
zs = zs[mm.active_set]
rs = rs[mm.active_set]

cnv = {i_full:i_act for i_act, i_full in enumerate(mm.active_set)}
dtrajs_act = []
pdb_inds = set()
n_events = 0
for i_traj, dtraj in enumerate(dtrajs):
    #--- search for splitting due to non-active states
    inds = []
    for i_ind, d_ind in enumerate(dtraj):
        if d_ind not in mm.active_set:
            inds.append(i_ind)
    for i_traj_bit, dtraj_bit in enumerate(np.split(dtraj, inds)):
        if i_traj_bit != 0: # otherwise the splitting index is included, i.e [:i_ind] will not include i_ind but [i_ind:] will
            dtraj_bit = dtraj_bit[1:]
        if len(dtraj_bit):
            dtraj_act = [cnv[d_ind] for d_ind in dtraj_bit] # converting to the index in active state
            dtrajs_act.append(dtraj_act) # so we should have more than, for ex., 500 trajectories if 10 sims and 50 gases because of splitting
            rst = rs[dtraj_act]
            if np.min(rst) < rmin_site:
                f = plt.figure()
                ax = f.add_subplot(1,1,1)
                ax.plot(dt*np.arange(len(dtraj_act)), rst, '-b')
                ax.plot(dt*np.arange(len(dtraj_act)), rmin_site*np.ones(len(dtraj_act)), '-r')
                ax.plot(dt*np.arange(len(dtraj_act)), rmax_site*np.ones(len(dtraj_act)), '-r')
                plt.xlabel('Time [ns]')
                plt.ylabel('Radius [A]')
                pdf.savefig()
                plt.ylim([0,40])
                print("traj: {}, split {}".format(i_traj,i_traj_bit+1))
                plt.title("traj: {}, split {}".format(i_traj,i_traj_bit+1))
                plt.close()
                intervals = []
                for ind in np.argsort(rst):
                    if rst[ind] >= rmin_site:
                        break
                    done = False
                    for interval in intervals:
                        if ind >= interval[0] and ind <= interval[1]:
                            done = True
                    if done:
                        continue
                    inds_left = np.arange(ind).astype(int)[rst[:ind] > rmax_site]
                    if len(inds_left):
                        ind_left = inds_left[-1]-1
                    else:
                        ind_left = 0
                    inds_right = np.arange(ind, len(rst)).astype(int)[rst[ind:] > rmax_site]
                    if len(inds_right):
                        ind_right = inds_right[0]+1
                    else:
                        #ind_right = 0
                        ind_right = len(rst)
                    intervals.append([ind_left, ind_right])
                    f = plt.figure()
                    ax = f.add_subplot(1,1,1)
                    ax.plot(dt*np.arange(len(dtraj_act[ind_left:ind_right])), rst[ind_left:ind_right], '-ob')
                    ax.plot(dt*np.arange(len(dtraj_act[ind_left:ind_right])), rmin_site*np.ones(len(dtraj_act[ind_left:ind_right])), '-r')
                    ax.plot(dt*np.arange(len(dtraj_act[ind_left:ind_right])), rmax_site*np.ones(len(dtraj_act[ind_left:ind_right])), '-r')
                    n_events += 1
                    for i, dind in enumerate(dtraj_act[ind_left:ind_right]):
                        pdb_inds.add(dind)
                    plt.xlabel('Time [ns]')
                    plt.ylabel('Radius [A]')
                    plt.title('left = {}, right = {}'.format(ind_left, ind_right))
                    pdf.savefig()
                    plt.ylim([0,40])
                    plt.close()
print('Number of events in {}/{}/{}: {}'.format(name, rmin_site, rmax_site, n_events))
write_pdb(range(len(mm.pi)), mm.pi, 'B')
if len(pdb_inds):
    write_pdb(np.array(list(pdb_inds)).astype(int), mm.pi, 'C')

with open('msm_{}.pk'.format(suffix), 'wb') as fout:
    pickle.dump(dtrajs_act, fout)
    pickle.dump(best_lag, fout)
    pickle.dump(dt, fout)
    pickle.dump(nits, fout)
    pickle.dump(box_min, fout)
    pickle.dump(box_max, fout)
    pickle.dump(dbox, fout)
    pickle.dump(xs, fout)
    pickle.dump(ys, fout)
    pickle.dump(zs, fout)
    pickle.dump(rs, fout)
    pickle.dump(mm, fout)

fpdb.close()
pdf.close()
