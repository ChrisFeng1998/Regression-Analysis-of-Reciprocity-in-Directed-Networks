
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
generate_data <- function(n,a,b,seed) {
  set.seed(seed)
  beta<-runif(n);beta<-beta-mean(beta);alpha<-runif(n);alpha<-alpha-mean(alpha)
  V1<-V2<-matrix(0,n,n)
  V1[upper.tri(V1)]<-runif(n*(n-1)/2,-1,1);V1<-V1+t(V1)
  V2[upper.tri(V2)]<-rnorm(n*(n-1)/2);V2<-V2+t(V2)
  mu<--a*log(n)+.5;theta<-c(b*log(n)+.5,1,1);A<-matrix(0L,n,n)
  for(i in 1:(n-1))for(j in (i+1):n) {
    eta<-c(0,mu+alpha[i]+beta[j],mu+alpha[j]+beta[i],
      2*mu+alpha[i]+beta[j]+alpha[j]+beta[i]+theta[1]+V1[i,j]+V2[i,j])
    w<-exp(eta-max(eta));s<-sample.int(4,1,prob=w)
    A[i,j]<-as.integer(s%in%c(2,4));A[j,i]<-as.integer(s%in%c(3,4))
  }
  list(A=A,V1=V1,V2=V2,theta=theta,n=n,a=a,b=b,seed=seed,alpha=alpha,beta=beta,mu=mu)
}

args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=5L) stop("Arguments: n a b data_seed output_binary")
d <- generate_data(as.integer(args[1]),as.numeric(args[2]),as.numeric(args[3]),as.integer(args[4]))
con <- file(args[5],"wb")
writeBin(as.raw(as.integer(t(d$A))),con)
writeBin(as.double(t(d$V1)),con,size=8,endian="little")
writeBin(as.double(t(d$V2)),con,size=8,endian="little")
close(con)
