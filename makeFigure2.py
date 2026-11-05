"""Schematic stellar-spectrum error budget, inspired by Rotman et al. Fig. 2.

Run: python make_stellar_error_figure.py --outdir .
Requires numpy and matplotlib. All amplitudes are illustrative, not measured.
Independent, zero-mean Gaussian latent terms are assumed, conditional on the
stellar mean spectrum and covariance hyperparameters. Calibration/extraction
terms here have already been linearized into fractional-flux units.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
COLORS = ['#526174', '#007F86', '#DC8A1A', '#5E5DC7',
          '#9C6644', '#2C82C9', '#B8497C', '#59933D']

def se(x, ell):
    return np.exp(-0.5*((x[:, None]-x[None, :])/ell)**2)

def draw(cov, rng):
    # Eigen sampling also handles singular low-rank nuisance covariances.
    eig, vec = np.linalg.eigh(cov)
    assert eig.min() > -1e-8, 'Covariance must be positive semidefinite'
    return vec @ (np.sqrt(np.maximum(eig, 0))*rng.normal(size=len(eig)))

def build_terms(lam):
    x = (lam-lam.min())/np.ptp(lam)
    g = lambda c, w: np.exp(-0.5*((lam-c)/w)**2)
    # A toy mean spectrum: band locations are illustrative, not identifications.
    mean = (lam/1.4)**-1.1 * (1-.22*g(1.38,.075)-.17*g(1.90,.10)-.12*g(2.32,.06))
    sigma = .16*(1+.45*(2*x-1)**2)
    noise = np.diag(sigma**2)
    cal = .38**2*se(lam,.65) + .16**2*np.ones((len(lam),len(lam)))
    extraction = .32**2*se(lam,.18) + .42**2*np.outer(g(1.15,.08),g(1.15,.08))
    # Linearized wavelength offset: derivative-shaped residual; covariance may
    # have negative entries, while remaining positive semidefinite.
    shift = np.gradient(np.log(mean),lam)
    shift = .35*shift/np.max(np.abs(shift))
    resolution = np.gradient(np.gradient(mean,lam),lam)/mean
    resolution = .18*resolution/np.max(np.abs(resolution))
    wave = np.outer(shift,shift)+np.outer(resolution,resolution)
    # An uncertain background level is additive in absolute flux. Dividing by
    # the mean gives this fractional-flux basis (stronger where flux is low).
    bg = .24*np.median(mean)/mean
    background = np.outer(bg,bg)
    env = .13+.17*g(1.40,.16)+.10*g(2.30,.15)
    interpolation = np.outer(env,env)*se(lam,.045)
    atmosphere = sum(a*a*np.outer(g(c,w),g(c,w)) for a,c,w in
                     [(.65,1.38,.075),(.55,1.90,.10),(.45,2.32,.06)])
    spot = .40*(1.2-x+.50*g(1.38,.10)+.35*g(1.90,.12))
    astrophysical = np.outer(spot,spot)
    return mean, [noise,cal,extraction,wave,background,interpolation,atmosphere,astrophysical]

def main(outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(2)  # Fixed illustrative realization.
    lam = np.linspace(.65,2.8,150)
    mean, terms = build_terms(lam)
    residuals = [draw(c,rng) for c in terms]
    total = sum(terms)
    combined = sum(residuals)
    titles = ['Measurement noise','Flux calibration','Position / extraction',
              'Wavelength / resolution','Background residual','Grid interpolation',
              'Atmosphere discrepancy','Unmodeled astrophysics']
    tags = ['EXPOSURE','INSTRUMENT / MODE','VISIT / DETECTOR','EXPOSURE / MODE',
            'EXPOSURE','GRID LOCATION','SIMILAR STARS','STAR / EPOCH']
    descriptions = [
        'Independent bins here;\nextraction can correlate noise.',
        'Sensitivity / flux scale.\nBroad wavelength coherence.',
        'Trace, flat field, aperture.\nBroad + local effects possible.',
        'Shift / line-spread function.\nOpposite-sign lobes correlate.',
        'Additive flux offset;\nfractional effect varies with flux.',
        'Numerical approximation.\nValidate on withheld grid nodes.',
        'Opacity / line-strength errors.\nBands may share model bias.',
        'Spots, companions, extinction.\nFit explicitly when relevant.'
    ]
    fig = plt.figure(figsize=(18,14),facecolor='white')
    fig.text(.055,.971,'Where does a stellar-spectrum mismatch come from?',
             fontsize=23,weight='bold',color='#172B3A')
    fig.text(.055,.943,'A Gaussian error budget: example residuals, wavelength covariance, and sharing across observations',
             fontsize=13,color='#526174')
    outer = GridSpec(3,4,figure=fig,left=.055,right=.95,bottom=.13,top=.895,
                     height_ratios=[1,1,.95],wspace=.28,hspace=.47)
    norm = TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
    cmap = 'RdBu_r'
    for k, (title,cov,resid) in enumerate(zip(titles,terms,residuals)):
        gs = GridSpecFromSubplotSpec(2,2,subplot_spec=outer[k//4,k%4],
                                    height_ratios=[.9,1.1],width_ratios=[1,1.13],
                                    hspace=.36,wspace=.18)
        ax = fig.add_subplot(gs[0,:])
        ax.set_title(f'{chr(97+k)}  {title}',loc='left',fontsize=12,weight='bold',pad=25)
        ax.text(0,1.12,tags[k],transform=ax.transAxes,color=COLORS[k],fontsize=8.5,weight='bold')
        ax.axhline(0,color='#CBD2D9',lw=.8)
        ax.fill_between(lam,-np.sqrt(np.diag(cov)),np.sqrt(np.diag(cov)),color=COLORS[k],alpha=.13,lw=0)
        ax.plot(lam,resid,color=COLORS[k],lw=1.3)
        ax.set(xlim=(.65,2.8),ylim=(-1.7,1.7),xticks=[.7,1.4,2.1,2.8],yticks=[-1,0,1])
        ax.tick_params(labelsize=8,length=3)
        ax.set_ylabel('Residual (%)',fontsize=8)
        ax.set_xlabel(r'$\lambda$ ($\mu$m)',fontsize=8,labelpad=1)
        mat = fig.add_subplot(gs[1,0])
        mat.imshow(cov/np.max(np.abs(cov)),origin='lower',extent=[.65,2.8,.65,2.8],
                   cmap=cmap,norm=norm,interpolation='nearest',aspect='equal')
        mat.set(xticks=[.7,2.8],yticks=[.7,2.8])
        mat.tick_params(labelsize=7,length=2)
        mat.set_xlabel(r'$\lambda_j$ ($\mu$m)',fontsize=8,labelpad=0)
        mat.set_ylabel(r'$\lambda_i$ ($\mu$m)',fontsize=8,labelpad=0)
        txt = fig.add_subplot(gs[1,1]); txt.axis('off')
        rms = np.sqrt(np.mean(np.diag(cov)))
        txt.text(0,.95,descriptions[k],va='top',fontsize=8.5,linespacing=1.45,color='#344454')
        txt.text(0,.10,f'RMS scale: {rms:.2f}%',fontsize=8.5,color=COLORS[k],weight='bold')

    ax = fig.add_subplot(outer[2,:2])
    ax.set_title('i  Independent latent terms add in covariance',loc='left',weight='bold',fontsize=13,pad=15)
    ax.fill_between(lam,-np.sqrt(np.diag(total)),np.sqrt(np.diag(total)),color='#D7E1EB',
                    label=r'Total marginal $\pm1\sigma$ (not a simultaneous band)')
    ax.plot(lam,combined,color='#192E42',lw=1.5,label='Sum of the eight example residuals')
    ax.axhline(0,color='#7D8B97',lw=.7)
    ax.set(xlim=(.65,2.8),xlabel=r'Wavelength ($\mu$m)',ylabel='Fractional residual (%)',ylim=(-3.3,3.3))
    ax.legend(loc='upper right',frameon=False,fontsize=8.5)
    ax.text(.02,.07,r'$r=\sum_k r_k,\qquad C_{\rm total}=\sum_k C_k$',
            transform=ax.transAxes,fontsize=13,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))

    gs = GridSpecFromSubplotSpec(1,2,subplot_spec=outer[2,2:],width_ratios=[1.05,1],wspace=.25)
    ax = fig.add_subplot(gs[0,0])
    ax.set_title('j  Sharing across stars / visits',loc='left',weight='bold',fontsize=13,pad=15)
    # Same wavelength sampling per block. Two visits each of three stars.
    # The rest-frame and observed-frame grids are taken identical in this toy.
    ix = np.linspace(0,len(lam)-1,35).astype(int)
    small = [c[np.ix_(ix,ix)] for c in terms]
    n = len(ix); joint=np.zeros((6*n,6*n))
    temps=np.array([3500,3800,5500])
    for a in range(6):
        for b in range(6):
            sa,sb=a//2,b//2
            block=small[1].copy() # same instrument calibration for all
            block+=np.exp(-.5*((temps[sa]-temps[sb])/650)**2)*small[6]
            if sa==sb:
                block+=small[5]+small[7] # same grid point and stable star term
            if a==b:
                block+=small[0]+small[2]+small[3]+small[4]
            joint[a*n:(a+1)*n,b*n:(b+1)*n]=block
    assert np.linalg.eigvalsh(joint).min()>-1e-8
    ax.imshow(joint/np.max(np.abs(joint)),origin='lower',cmap=cmap,norm=norm,interpolation='nearest')
    centers=np.arange(6)*n+(n-1)/2
    labels=['A1','A2','B1','B2','C1','C2']
    ax.set(xticks=centers,xticklabels=labels,yticks=centers,yticklabels=labels)
    ax.tick_params(length=0,labelsize=8)
    for bound in np.arange(1,6)*n-.5:
        ax.axhline(bound,color='white',lw=.7);ax.axvline(bound,color='white',lw=.7)
    ax.set_xlabel('Star + visit; wavelength within each block',fontsize=8)
    txt=fig.add_subplot(gs[0,1]);txt.axis('off')
    txt.text(0,.97,'Off-diagonal blocks matter',weight='bold',fontsize=10,va='top')
    txt.text(0,.82,'A1 / A2: repeat visits of star A\nB1 / B2: a similar star\nC1 / C2: a hotter star\n\nCalibration links every block.\nAtmosphere error links similar stars.\nVisit errors stay in diagonal blocks.\n\nShared errors do not average down\nas independent measurement noise.',
             va='top',fontsize=9,linespacing=1.55,color='#344454')

    cax=fig.add_axes([.962,.42,.009,.22])
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=[-1,0,1])
    cb.set_label(r'Covariance / max $|C_{ij}|$ (each matrix)',fontsize=9,labelpad=8)
    cb.ax.tick_params(labelsize=8)
    fig.text(.055,.076,'How to read: lines = one Gaussian draw; shading = marginal ±1σ; matrices = covariance shape, each scaled by its own maximum.',fontsize=10,color='#344454')
    fig.text(.055,.052,'Schematic amplitudes, band locations, and sharing assumptions. Error sources are not uniquely identifiable from residual shape alone.',fontsize=10,color='#344454')
    fig.text(.055,.028,'Inspired by the additive-kernel illustration in Rotman et al. (2025), Fig. 2, arXiv:2503.21702. New synthetic stellar-spectrum demonstration.',fontsize=9,color='#526174')
    for ext in ['png','pdf','svg']:
        fig.savefig(outdir/f'stellar_error_budget.{ext}',dpi=220,facecolor='white')
    plt.close(fig)
    np.savez_compressed(outdir/'schematic_covariances.npz',wavelength_um=lam,
                        mean_spectrum_arbitrary=mean,component_covariances_percent2=np.array(terms),
                        component_residuals_percent=np.array(residuals),total_covariance_percent2=total,
                        joint_covariance_percent2=joint,component_names=np.array(titles))
    print(f'Created PNG, PDF, SVG, and covariance data in {outdir.resolve()}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=Path(__file__).resolve().parent)
    main("Figures4Paper")