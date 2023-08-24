##### !/usr/bin/env python
# coding: utf-8

# In[3]:
import warnings
warnings.filterwarnings('ignore')


import numpy as np
from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer import LikelihoodComputationChain
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras as ks
from cosmoHammer import CosmoHammerSampler
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras as ks
#import MCMC_CosmoHammer as mcmc
import time
from cosmoHammer.util import Params
#peak, min., max., jump
from ann_input import *
# In[2]:
from math import *
import numpy as np
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.facecolor']='white'
#pi=math.pi

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[10]:

noise=np.loadtxt('data/noise')
prior= Params(('Mhmin',[5.0,0.0730406,158.2789802,0.00730406]),
              ('alpha', [1.58,1.58,1.58,0]))
err=np.loadtxt('data/cii_er')
par=np.loadtxt('data/params_LH.txt')
mh=np.mean(par[:,0])

ms=np.std(par[:,0])

#Transforming and inverse transforming parameters
def tr(p):
    y=((p-[mh,0])/[ms,1])/4
    return y

def inv_tr(p):
    x=4*np.multiply([ms,1],p)+[mh,0]
    return x

class Core_Module(object):
    def __init__(self,model_name):
        self.model=ks.models.load_model(model_name)

    def __call__(self,ctx):
        par=ctx.getParams()

        ctx.add('params',par)

        params = np.array([i for i in par])
        params=np.reshape(tr(params),(1,2))
        model = ks.models.load_model('cii_model.h5')

        model_th=model.predict(params, verbose=0)
        # model_th=np.exp(10*model_th)

        ctx.add("model_th",model_th)


    @staticmethod
    def setup():
        print("Core is done!")


# In[17]:


class Likelihood_Module(object):
    def __init__(self,data,nbins,noise):
        self.data= data
        try:
            eye = np.eye(len(data))
        except:
            eye=np.eye(1)
        if np.sum(nbins) != 0.:
            cov = abs(data**2) / nbins
            cov = cov + np.abs(noise**2)
            cov = eye * cov
            cov_inv = np.linalg.inv(cov)

        else:
            cov = np.abs(data/np.sqrt(nbins))+noise
            cov = eye * cov
            cov_inv = np.linalg.inv(cov)

        self.div = 1.0
        self.cov = cov
        self.cov_inv = cov_inv
    
    #chi-square likelihood
    # def computeLikelihood(self,ctx):
    #     model_th=ctx.get('model_th')
    #     #RAGHU# in our case model PS will be estimated using the emulator.
    #     # the likelihood is sum of the lot of normal distributions
    #     eps = self.cov
    #     denom = np.power(eps,2)
    #     lp = -0.5*sum(np.divide(np.power((self.data - model_th),2),denom))
    #     return lp
    
    #original likelihood
    def computeLikelihood(self,ctx):
        model_th=ctx.get('model_th')
        diff=np.subtract(model_th,self.data).reshape(1,len(self.data))
        logl=-np.dot(diff,np.dot(self.cov_inv,diff.T))/2.
        return logl
    
    #MSE LIKELIHOOD
    # def computeLikelihood(self, ctx):
    #         model_th = ctx.get('model_th')
    #         diff = np.subtract(model_th, self.data).reshape([1, -1])
    #         mse = np.mean(np.square(diff))
    #         logl = -mse / 2.0
    #         return logl

    @staticmethod
    def setup():
        print("Likelihood setup done!")


# In[18]:


class RunMCMC:
    """ sampler & MPI sampler class """

    def __init__(self, prior, data, nbins, model, noise=0., div=1.0, like_func='n'):
        """
        :param data: load your data
        :param nbins: number of k-modes in powerspectrum OR
         number of triangle contributions in bispectrum (for covariance matrix)
        :param noise: system noise, e.g. SKA, MWA noise response (if any), default 0.0,
        :param div: likelihood normalization factor, default 1.0,
        :param like_func: choose between complex likelihood function (use 'c'), and normal function (use 'n')
        prefer complex likelihood for bispectrum
        """
        self.params=prior
        chain = LikelihoodComputationChain(min=self.params[:, 1], max=self.params[:, 2])
        chain.params = prior
        if like_func == 'n':
            chain.addLikelihoodModule(Likelihood_Module(data,nbins,noise))
        else:
            chain.addLikelihoodModule(ComplexLikeModule(data, nbins, noise, div))
        self.chain = chain


    def load_model(self, load_model='cii_model.h5', name='pk'):

        """
        :param load_model: load your own model, (give the path)
        :param name: name for data, ('pk','bk')==>for powerspectrum, bispectrum
        :param norm: rescale used in the training
        """
        self.name = name
        self.chain.addCoreModule(Core_Module(load_model))
        self.chain.setup()


    def sampler(self, walker_ratio, burnin, samples, num, threads=-1):
        """
            :param walker_ratio:  the ratio of walkers and the count of sampled parameters
            :param burnin: burin iterations
            :param samples: no. of sample iterations
            :param num: number to put in output files e.g: string(name+num)=Pk_1,Bk_1
            :param threads: no. of cpu threads

            self.chain.setup()
            print("find best fit point")
            pso = MpiParticleSwarmOptimizer(self.chain, params[:, 1], params[:, 2])
            psoTrace = np.array([pso.gbest.position.copy() for _ in pso.sample()])
            params[:, 0] = pso.gbest.position
        """


        sampler = CosmoHammerSampler(
            params=self.params,
            likelihoodComputationChain=self.chain,
            filePrefix='%s' % self.name + '%d' % num,
            walkersRatio=walker_ratio,
            burninIterations=burnin,
            sampleIterations=samples, threadCount=threads)


        print("started sampling:")
        start = time.time()
        sampler.startSampling()
        end = time.time()
        tics = end - start
        print("The time taken %.2f sec. done!" % tics)
        print('Done!')



pk_test=np.loadtxt('data/cii_dpk')
n=np.loadtxt('data/nbins.txt')
params_test=np.loadtxt('data/params_LH.txt')

#i=np.random.randint(low=0,high=len(params_test),size=2)
i=[83,183,202]
fn_t=pk_test[i]

print("Chosen  indices: ",i)
print("chosen values: ",params_test[i], "\n Chosen pk",pk_test[i])

samples=10000
w=2

for el in range(len(i)):
    st=time.time()
    sampler=RunMCMC(prior=prior,data=fn_t[el],nbins=n,noise=noise,model='noise')
    sampler.load_model()
    sampler.sampler(walker_ratio=w,burnin=0.1*samples,samples=samples,num=i[el])
    tt=time.time()-st
    print("Time taken for ",i[el],time.strftime("%H-%M-%S",time.gmtime(tt)))

#plotting the results
for el in range(len(i)):
    data=np.loadtxt('pk'+str(i[el])+'.out') #sample saved with name 'model num.out'

    truth = [params_test[i[el],0]]

    c = ChainConsumer()
    c.add_chain(data[:,0], parameters=[r"$M_{(h, {\rm min})}(\rm 10^{10} M_\odot)$"],
                    name='CV+Noise',color='#F28482')
    c.configure(label_font_size=18,linestyles='-',linewidths=2, tick_font_size=18,shade_alpha=1, )
    c.configure_truth(color='k', ls=":", lw=1.5)
    
    fig = c.plotter.plot(truth=truth)
    fig.set_size_inches(5 + fig.get_size_inches())
    plt.savefig('plot_'+'{:.2f}'.format(truth[0])+'_emulated.png',bbox_inches='tight',dpi=100)
