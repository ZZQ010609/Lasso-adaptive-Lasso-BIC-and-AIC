# GCV function
GCV = function(x, y, b, sigmahat){
  g = sum((x%*%b-y)^2) /
    (1 - sum(diag(x%*%solve(t(x)%*%x+n*sigmahat)%*%t(x)))/n)^2
  return(g)
}


# beta_lasso=sgn(beta)(|beta|-lambda)+

rela = function(bls, lambda){
  a = bls/abs(bls) 
  b = abs(bls)-lambda 
  b[which(b<0)] = 0 
  beta = a*b 
  return(beta)
}

# lasso
lasso = function(x,y){ 
  n = length(y) 
  bls = solve(t(x)%*%x)%*%t(x)%*%y #OLS
  gcv = c() 
  k = 1 #the subscript of GCV
  # loop to find lambdahat:
  for (lam in seq(0.01,0.4,0.01)) { 
    b0 = bls #change the initial value of lambda 
    #iterative： 
    repeat{ b0[c(which(abs(b0)<0.0001))] = 10e-8 #when beta= 0  sigmahat equals large number 
    sigmahat = diag(as.numeric(lam/abs(b0))) 
    b0[which(b0==0.0001)] = 0 # let too small value be 0
    U = 2*t(x)%*%(x%*%b0-y) + n*sigmahat%*%b0 
    U1 = t(x)%*%x + n*sigmahat 
    b1 = b0 - solve(U1) %*% U 
    if (sqrt(sum((b0-b1)^2))<0.001) 
      break # stop iterating when converge
    b0 = b1 #the initial value for next iteration }
    #stop iteration，record the results： 
    gcv[k] = GCV(x, y, b0, sigmahat) # the gcv of the kth lambda
    k = k+1 }
    lambda = seq(0.01,0.4,0.01)[which.min(gcv)] 
    beta = rela(bls,lambda) 
    return(beta) 
  }
  
  # adaptive lasso

  adlasso = function(x, y){ 
    n = length(y) 
    bls = solve(t(x)%*%x)%*%t(x)%*%y 
    w = 1/abs(bls) #weight 
    gcv = c() k = 1 #the subscript of GCV and betahat 
    #loop to find lambdahat： 
    for (l in seq(0.01,0.3,0.01)) { 
      lam = l*w 
      b0 = bls 
      #start to iterate
      repeat{ b0[c(which(abs(b0)<0.0001))] = 0.0001 
      #when coefficients equals 0, sigmahat equal large number
      sigmahat = diag(as.numeric(lam/abs(b0))) 
      b0[which(b0==0.0001)] = 0 #when beta is small, let beta be 0
      U = 2*t(x)%*%(x%*%b0-y) + n*sigmahat%*%b0 
      U1 = t(x)%*%x + n*sigmahat 
      b1 = b0 - solve(U1) %*% U 
      if (sqrt(sum((b0-b1)^2))<0.001) 
        break #stop iterating
      b0 = b1
      }
      #record results 
      gcv[k] = GCV(x, y, b0, sigmahat) 
      gcv k = k+1 
    }
    lambda = seq(0.01,0.3,0.01)[which.min(gcv)]*w 
    beta = rela(bls,lambda) 
    return(beta) 
  }
  # the label matrix of 63 subsets of explantory variables 
  Select = function(){ 
    set = matrix(NA, 2^6-1, 6) 
    tag = 1  #the order of 63 subsets, Mainly used when filling the set matrix
    for (a in 0:1) { 
      for (b in c(0,2)) { 
        for (c in c(0,3)) { 
          for (d in c(0,4)) { 
            for (e in c(0,5)) { 
              for (f in c(0,6)) { 
                select = c(a, b, c, d, e, f) 
                if (sum(select) == 0) 
                  next # Skip the case with no explanatory variables
                set[tag, ] = select #The selection at this time is recorded in the tag line
                tag = tag + 1 #the number of Subset + 1
              } 
            } 
          } 
        }
      } 
    }
    return(set) 
  }
  set = Select()
  
  # the function of best subset selection，output the estimator of beta with critia AIC and BIC 
  fullsubset = function(X, y){ 
    n = length(y) #sample size
    b = matrix(0, 63, 6) #A matrix recording the results of parameter estimates for all subsets
    aic=c() 
    bic=c() 
    for (i in 1:63) { #The ith subset selection scheme is the ith row of set
      x = as.matrix(X[,set[i,]]) #reset explanatory variables
      b[i, set[i,]] = solve(t(x)%*%x)%*%t(x)%*%y #Parameter Estimation
      sse = sum((x%*%b[i,set[i,]]-y)^2) #the sum of residuals
      p = dim(x)[2] #the number of the parameter
      aic[i] = n*log(sse)+2*p #calculate AIC
      bic[i] = log(sse/n)+p/n*log(n) #Calculate BIC
    }
    baic = b[which.min(aic), ] #AIC selection result
    bbic = b[which.min(bic), ] #BIC selection result 
    beta = rbind(baic, bbic)
    return(beta) 
  }
  
  
  # generate random numbers
  library(MASS) 
  generate = function(n, sigma, sig){ 
    x = mvrnorm(n, rep(0,6), sigma) 
    eps = rnorm(n, 0, sig) #esp~N(0,sig^2) 
    beta0 = c(1, 0.8, 0.6, 0, 0, 0) 
    y = x%*%beta0 + eps 
    data = data.frame(x,y) return(data) 
  }
  
  rate = function(B, betahat){ 
    B=1000  #do the simulation for 1000 times 
    correct = sum(betahat[,4:6]==0)/B 
    incorrect = sum(betahat[,1:3]==0)/B 
    r = c(correct, incorrect) 
    return(r) 
  }
  
  
  # output the result form
  result = function(B, n, sigma, sig){ 
    Las = matrix(NA, B, 6) 
    Adl = matrix(NA, B, 6) Aic = matrix(NA, B, 6) 
    Bic = matrix(NA, B, 6) 
    for(i in 1:B){ 
      data = generate(n, sigma, sig) 
      y = data$y 
      x = as.matrix(data[,1:6]) 
      Las[i,] = lasso(x, y) 
      Adl[i,] = adlasso(x, y) 
      full = fullsubset(x, y) 
      Aic[i,] = full[1,] 
      Bic[i,] = full[2,] 
    }
    LASSO = rate(B, Las) 
    AdaLASSO = rate(B, Adl) 
    AIC = rate(B, Aic) 
    BIC = rate(B, Bic) 
    res = rbind(LASSO, AdaLASSO, AIC, BIC) 
    colnames(res) = c("correct rate", "incorrect rate") 
    return(res) 
  }
  
  
  # run the code
  result(1000, 100, diag(1,6), 2)