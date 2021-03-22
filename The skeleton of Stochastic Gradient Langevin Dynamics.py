#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def mini_batch(Num_sample,batch_size):
  '''
  Inputs:
    Num_sample: number of total samples
    batch_size: the length of the mini_batch
  ------------------------------------------------
  outputs:
    list_mini_batch: list containing mini_batches' lists
  '''
  L=list(np.arange(Num_sample))
  list_mini_batch=[]
  while len(L)>batch_size:  
    batch=list(np.random.choice(L, batch_size, replace=False))
    list_mini_batch.append(batch)
    res = [i for i in L if i not in batch]
    L=res
  if L!=[]:
    list_mini_batch.append(L)
  return(list_mini_batch)


def prior_grad(theta):
  '''
  This function computes the gradient of the prior funtion at theta
  '''
  return(1)

def lk_hood_grad(x,theta):
  '''
  This function computes the gradient of the likelihood of x for the parameter theta
  '''
  return(1)

def eps_t(t):
  ''' 
  this function computes the value of the rate epsilon at instant t
  '''
  return(1)

def step_update(theta,mini_batch,data):
    likelihood_grad=sum([lk_hood_grad(data[i],theta) for i in mini_batch])*(len(data)/len(mini_batch))
    new_step=prior_grad(theta) + likelihood_grad
    return(new_step)


def Stoc_lang_grad(data,batch_size,len_feature,num_repet,num_run):
  '''
  This function applies the Stochastic Gradient Langevin Dynamics
  ----------------------------------------------------------------
  inputs:
   data:
   batch size: integer
   num_repet: number of sweeps through the whole data
   num_run: number of run 
   len_feature: the legth of the learned vector
  ----------------------------------------------------------------
  outputs: list of the learned vector during iterations

  '''
  Num_sample=len(data)
  theta_final_list=[]
  for run in tqdm(range(num_run)):
    list_mini_batch=mini_batch(Num_sample,batch_size)
    j=0
    theta_list=[np.random.rand(len_feature)]
    for i in range(1,num_repet* len(list_mini_batch)+1):
      theta= theta_list[-1] + (0.5*eps_t(i)*(step_update(theta_list[-1],list_mini_batch[j],data)+ np.random.normal(0,sqrt(eps_t(i)),size=len_feature)))
      if j< (len(list_mini_batch)-1):
        j+=1
      else:
        j=0
      theta_list.append(theta)
    theta_final_list.append(theta_list)
  return(np.sum(theta_final_list,axis=0)/num_run)

