from examples.tvb_nest.notebooks.cerebellum.scripts.scripts import *
import numpy
import sys
config, plotter = configure()
def Calc_Zs_Shrink_param(param,iG,TrainSamples):
    foldername ='/home/docker/packages/tvb-multiscale/examples/tvb_nest/notebooks/cerebellum/outputs/cwc/res/'
    folderpath = os.path.join(".", foldername)
    bpsfolderpath = os.path.join(folderpath, "bps")
    bpsfilepath = lambda iB: batch_priors_filepath(iB, config, iG, filepath=bpsfolderpath, extension=".pt")
    priors_samples = []
    for iB in range(0, 500):
        priors_samples.append(torch.load(bpsfilepath(iB)))
    priors_samples = torch.concat(priors_samples)    
    config.PRIORS_DEF[param]['min']
    config.PRIORS_DEF[param]['max']
    prior_var=((config.PRIORS_DEF[param]['max']-config.PRIORS_DEF[param]['min'])**2)/12
    #fig,axs=plt.subplots(5,1)
    Zg=[]
    Sg=[]
    Zgst=[]
    for j in range(len(TrainSamples)):
        n_train_samples=TrainSamples[j]
        filename = "samples_fit_iG%02d_%04d_Train.npy" % (iG,n_train_samples)
        filepath = os.path.join(config.out.FOLDER_RES, filename)
        Postfile =np.load(filepath,allow_pickle=True)
        Testsamples=np.shape(Postfile.item()['samples'])
        #print("number of Testsamples:" ,Testsamples[0])
        Z=[]
        S=[]
        if param == 'I_s':
            p=0
        elif param == 'FIC':
            p=1
        elif param =='FIC_SPLIT':
            p=2
        else:
            raise ValueError("Input Argument should be :I_s | FIC | FIC_SPLI")

        for i in range(Testsamples[0]):
            z=(priors_samples[i][p]-Postfile.item()['mean'][i][p])/Postfile.item()['std'][i][p]
            s=1-(numpy.power(Postfile.item()['std'][i][p],2)/prior_var)
            Z.append(z)
            S.append(s)
        Zgst.append(numpy.std(Z))
        Zg.append(numpy.mean(np.abs(Z)))
        Sg.append(numpy.mean(S))
    
    #plt.plot(TrainSamples,Zg)
    
    #plt.title('MeanZ-TrainSple_G%g_%s' % (Postfile.item()['G'],param))
   # plt.xlabel('N_Train_Samples')
    #plt.ylabel('Zscore_Mean')
    #plt.savefig(os.path.join(config.figures.FOLDER_FIGURES, 'TrainSamples_Zscore_Mean_G%g_%s.png' % (Postfile.item()['G'], param)))
    #plt.close()
    return Zg,Sg,Zgst
    
if __name__ == "__main__":
    Default={'I_s': 0.08,'FIC': config.FIC,'FIC_SPLIT': config.FIC_SPLIT}
    colors=['b','g','r','c','m','y','k','w','c']
    c=0
    TrainSamples=[100,200,250,300,350,400,450,500,600,650,700,750,800,850,900,950,1000,2000,3000,4000,5000]
    for iG in range(3):
        print('for iG:',iG)
        for k,v in Default.items():
            param=k
            
            Zg,Sg,Zgst=Calc_Zs_Shrink_param(param,iG,TrainSamples)
            print('for param :',param)
            print('Zscore :',Zg)
            print('Shrinkage:',Sg)
            print('Zscore std :',Zgst)
            plt.plot(TrainSamples,Zg,color=colors[c] ,label='Zscore_iG%g_%s'%(iG,param))
            plt.legend(loc='upper left')
            c=c+1
            #plt.plot(TrianSamples,Sg,color,label='Shrinkage_iG%g_%s'%(i
            plt.xlabel("TrainSamples")
            plt.ylabel("Zscore")
    plt.savefig(os.path.join(config.figures.FOLDER_FIGURES,'Zscore_IS_FIC_FICSPLIT_2fev.png'))             


