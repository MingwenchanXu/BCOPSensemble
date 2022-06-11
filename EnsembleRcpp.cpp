#include <Rcpp.h>
using namespace Rcpp;

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::export]]
double meanC(NumericVector x) {
  int n = x.size();
  double total = 0;
  
  for(int i = 0; i < n; ++i) {
    total += x[i];
  }
  return total / n;
}


// [[Rcpp::export]]
int conmformalCompare(NumericVector xte, NumericVector x) {
  int count = 0;
  int l = xte.size();
  for(int i = 0; i < l; i++){
    if(xte[i] >= x[i]){
      count += 1;
    }
  }
  
  return count;
}

// [[Rcpp::export]]
NumericVector CompareLoop( const int xteL,  
                           const NumericMatrix PredFull, 
                           const int rowK,
                           const NumericMatrix indexM, 
                           const NumericVector trainV,
                           const int B){
  NumericVector xte_sk = no_init_vector(xteL);
  
  for(int ite = 0; ite < xteL; ite++){
    NumericVector vk = no_init_vector(rowK);
    NumericVector scoreFull  = no_init_vector(B);;
    scoreFull = PredFull(ite, _);
    
    for(int i = 0; i < rowK; i++){
      NumericVector indexTemp  = indexM(i,_);
      indexTemp = na_omit(indexTemp);
      indexTemp = indexTemp-1;
      NumericVector scoreSelect = scoreFull[indexTemp];
      vk[i] = meanC(scoreSelect);
    }
    
    int numComp = conmformalCompare(vk,trainV);
    xte_sk[ite] = (1+numComp)/(rowK+1.0);
  }
  
  return xte_sk;
}



// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
# CompareLoop(xte.sk, l.row,test.pred.full,rowK, index.pred.k.matrix,train.vk)
# CompareLoop( 5, test.pred.full)
*/
