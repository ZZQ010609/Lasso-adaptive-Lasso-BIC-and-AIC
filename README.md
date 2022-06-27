# The Comparison of Variable Selection Methods Based on Simulation Study
## Motivation
There are many ways to do the variable selection, such as subset selection, compressed estimation, and dimensionality reduction. Subset selection is 
generally based on AIC, BIC, and adjusted R2; common methods for compression estimation include Lasso, Adaptive Lasso, ridge regression, and SCAD methods; 
dimensionality reduction includes principal component regression, partial least squares, etc. This paper will use R language to implement computer simulation, 
use the optimal subset selection method (selected according to AIC, BIC), Lasso, Adaptive Lasso for variable selection, and compare the correct error rate.

## Code
Now we want to do the simulation for 1000 times and output the parameter estimates 
of the four methods each time. Finally, all the estimtes are aggreated to calculate
estimated correct rate and error rate. The correct rate is the average number of 
correctly estimationg zero parameters as 0; the error rate is the average number of 
non-zero parameters estimated to be 0. In each simulation, we will generate 100 normal random numbers,
$X_{100 \times 6} \sim N(0,I_6), \epsilon_{100×1} \sim N(0,4),y_i=X_i \beta_0+\epsilon_i,\ where\  \beta_0=(1,0.8,0.6,0,0,0)'$. 
Then four methods will be used to estimate the data and the output results will be four sets
of parameter estimates. Now I will show the function of four methods.

### Function of four subset selection methods 
#### 2.1 Lasso
The cost function of Lasso is $Q(\beta)=\frac{1}{n}\sum (y_i-\hat{y_i})^2+\lambda\sum|\beta_k|$.
When $\lambda$ is fixed, I use LQA algorithm to get the esimated 
$\beta$. Then traverse $\lambda\ to\ find\ \hat{\lambda}$ that minimizes GCV. The algorithm process of LQA is as follows:

**Step 1** : Let the initial value for $\beta^{(0)}=(X'X)^{-1} X'y$ 

**Step 2** : iterate:

$β^{(1)}=β^{(0)}-[▽^2 Q_{LS}(β^{(0)})+nΣ_λ(β^{(0)})]^{(-1)} [▽Q_{LS} (β^{(0)})+nU_λ(β^{(0)})],\ where\ ▽^2 Q_{LS}(β^{(0)})=X'X;▽Q_{LS} (β^{(0)})=-2X'(Y-Xβ^{(0)}) $
$Σ_λ (β^{(0)})=λdiag(1/|β_0 |,...,1/|β_p |), U_λ (β^{(0)})=Σ_λ (β^{(0)})β^{(0)}$

If $β_j=0，then\ 1/|β_j |=10^8$

**Step 3**:

Repeat the above steps until the estimated $\beta\ converges.\ It\ is\ considered\ that\ when\ ‖\beta_k^{(0)} − \beta_k^{(1)}‖ ≤ 10^{−3}$, the iteration can be stopped.

Now we want to change the value of $\lambda$ and minimize the GCV.
Let the the grid of $\lambda$ from 0.01 to 0.4, and the length is 0.01
$GCV(λ)=\frac{|Y-Xβ_λ |^2}{(1-\frac{1}{n}tr(P_x(\lambda)))^2 },\ where\ P_x(\lambda)=X[X'X+n\sum(\hat{\beta_{\lambda}})]^{-1}X'$
$\lambda_{best}=argmin𝐺CV(\lambda),\ Then\ \beta_{lasso}=sgn(β_{λ_{best} })(|(β_{λ_{best} } ) ̂|-λ_{best})_+$

```ruby
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

```

#### 2.2 adaptive Lasso
The difference between the Lasso and adaptive Lasso is the penalty. The penalty of adaptive Lasso is
$\Sigma_{k=0}^p \lambda_k \beta_k,\ where\ \lambda_k={\lambda}/{\beta_K^{LS}} , \beta_K^{LS}$  is the estimator of OLS.
Therefore, $Σ_λ (β^{(0)})=λdiag(w_0/|β_1^{(0)} |,...,w_p/|β_p^{(0)} |),where\ w_k=1/|(β_k ) ̂^{LS} |$,
the estimator of optimal $beta_{adlasso}=sgn(β_{λ_best })(|(β_{λ_best } ) ̂|-λw_k )_+$

``` ruby
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
   
```

#### 2.3 Best Subset Selection (AIC and BIC)
This method is to let all subsets of all explanatory variables for fitting, and then select the optimal model with the minimum AIC and BIC.

The formula of AIC is AIC=ln(SSE)+2p/n. The formula of BIC is BIC=ln(SSE)+(pln(n))/n,  where SSE is the sum of the squared residual, p is the number of explantory variables, n is the sample size.

The key problem of the optimal subset selection method is how to traverse all the possibilities of subsets, and correspond the subsets with  AIC and BIC. I construct a label 
matrix where the ith element of each row that is not zero is the ith selected explanatory variable.

```ruby
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
```
### other functions
#### generate random numbers
```ruby

library(MASS) 
generate = function(n, sigma, sig){ 
    x = mvrnorm(n, rep(0,6), sigma) 
    eps = rnorm(n, 0, sig) #esp~N(0,sig^2) 
    beta0 = c(1, 0.8, 0.6, 0, 0, 0) 
    y = x%*%beta0 + eps 
    data = data.frame(x,y) return(data) 
}
```

#### #calculate correction rate and incorrect rate 
```ruby
rate = function(B, betahat){ 
  B=1000  #do the simulation for 1000 times 
  correct = sum(betahat[,4:6]==0)/B 
  incorrect = sum(betahat[,1:3]==0)/B 
  r = c(correct, incorrect) 
  return(r) 
}

#### output the result form
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

```

### run the code
```ruby
result(1000, 100, diag(1,6), 2)
```
