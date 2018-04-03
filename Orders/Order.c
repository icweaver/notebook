#include <Python.h>
#include <numpy/arrayobject.h>
#include <sys/time.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#define NormA 0.39894228 // 1/sqrt(2*pi)
#define ELEM_SWAP(a,b) { register double t=(a);(a)=(b);(b)=t; }
#define ARRAYD(p) ((double *) (((PyArrayObject *)p)->data)) 

/* 
 *                                [INITIALIZATION]
 * ------------------ PROTOTYPES FOR FUNCTIONS AND EXTERNAL VARIABLES -----------------------
 *
 */

double* DeleteElement(double* A,int Element,int rows);
double getSigma(double* Data,double* Model,int rows);
void SigmaClipping(double* vector,double* x,double* Data,double* Model,int rows);
double** OrderLocation(double *m,int *norders,int len_rows,int len_cols,int mid_col,int Window,double* Filter,int option);
int getMiddleColumn(int len_cols);
double quick_select(double arr[], int n);
double* MedianFilter(double* A,int len_A,int Window);
double MedianWindowing(double* A,int pos,int len_A,int Window);
double* MakeVector(int nelements);
double** MakeArray(int rows, int columns);
void FreeArray(double** theArray,int rows);
int CheckOrder(double* M,double Threshold,int i,int len_rows);
double MeanCalculator(double* M,int imin,int i);
double MAD(double* M,int imin,int i,double Mean);
int CheckMax(double* M,int i,int len_rows);
int CheckEndOrder(double* M,int i,int s,double Threshold,int Max,int len_rows);
double* PolynomialFit(double* data_x,double* data_y,int n,int order);
double* PolynomialEval(double* A,double* x,int order,int len_x);
void LinearSolver(double** A,double* b,double* x,int length);
double** FitDetection(double** A,int len_rows,int len_rows_image,int *norders);
double PCrossCorrelation(double *m,double Loc,double Ap,int CurrentColumn,int len_cols,double precision,double* Filter,int option,int len_rows);
double CCrossCorrelation(double *m,int len,double precision,double sigma);
double* WhereMax(double* Vector, int n);

static PyObject *Order_CCFCenter(PyObject *self, PyObject *args){
        double *m;
        double precision,sigma;
        int len;
        PyObject *marray;
        PyArg_ParseTuple(args,"Oidd",&marray,&len,&precision,&sigma);
        m = ARRAYD(marray);              
        double mu = CCrossCorrelation(&m[0],len,precision,sigma);
        return Py_BuildValue("d",mu);
}

/*                  [INITIALIZATION OF A METHOD]
*------------------------THE LOCATE METHOD-----------------------------
* 
* We use threshold-detection methods in order to obtain orders in different 
* columns.
*
*----------------------------------------------------------------------
*/

static PyObject *Order_Locate(PyObject *self, PyObject *args){
        struct timeval tim;
        gettimeofday(&tim, NULL);
        double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	double *m;
        int len_rows,len_cols,mid_col,Window,norders=0,option;
	PyObject *marray;
	
/* 
 *--------------------------------THE DATA---------------------------------------
 * After initialization of the PyObject pointers, we wish to recover the following inputs:
 *
 * marray:   Vector of the flattened-matrix of the Echelle Spectra data.
 *
 * len_rows: Length of the rows of the flattened-matrix.
 *
 * len_cols: Length of the columns of the flattened-matrix.
 *
 * mid_col:  Column where we'll measure the orders (-1 indicates the mid column).
 * ------------------------------------------------------------------------------
*/
	PyArg_ParseTuple(args,"Oiiiii",&marray,&len_rows,&len_cols,&mid_col,&Window,&option);
	m = ARRAYD(marray);                              /* We convert our PyObject struct pointer to a C-vector array.           */
        if(Window%2==0){
          printf("Input error: Only uneven numbers are allowed as windows. \n");
          return Py_BuildValue("i",0);
        }
        double* Filter = MakeVector(len_rows);
        double** Output = OrderLocation(&m[0],&norders,len_rows,len_cols,mid_col,Window,Filter,option);
        free(Filter);

/* Start of the matrix-to-vector conversion part of the code */    

        double* theArray;
        int i,j;
        theArray = (double*) malloc((norders*2+1)*sizeof(double));
        theArray[0]=norders;
        for(i=0;i<norders;i++){
           for(j=0;j<2;j++){
               theArray[i*2+j+1]=Output[i][j];
            } 
        }
/* End of the matrix-to-vector conversion part of the code.*/


/* Finally, we create a Python "Object" List that contains the P coefficients and return it back to Python */

        PyObject *lst = PyList_New(norders*2+1);
        if (!lst)
        return NULL;
        for (i = 0; i < norders*2+1; i++) {
          PyObject *num = PyFloat_FromDouble(theArray[i]);
          if (!num) {
          Py_DECREF(lst);
          return NULL;
          }
          PyList_SET_ITEM(lst, i, num);
        }
        gettimeofday(&tim, NULL);
        double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
        printf("%2.2lf seconds elapsed\n", t2-t1);
	PyObject *MyResult = Py_BuildValue("O",lst);
        Py_DECREF(lst);
        return MyResult;
}

/*                  [INITIALIZATION OF A METHOD]
*------------------------THE CENTERS METHOD-------------------------------
* 
* Here, given the located orders in the LOCATE method, we detect the centers
* on each column for each order using Cross-Correlation.
*
*-------------------------------------------------------------------------
*/

static PyObject *Order_getCenters(PyObject *self, PyObject *args){
        struct timeval tim;
        gettimeofday(&tim, NULL);
        double t1=tim.tv_sec+(tim.tv_usec/1000000.0),precision=0,Aperture,Center;
	double *m;
        int len_rows,len_cols,Norders,norders,Window,option,l;
	PyObject *marray;
	
/* 
 *--------------------------------THE DATA---------------------------------------
 * After initialization of the PyObject pointers, we wish to recover the following inputs:
 *
 * marray:   Vector of the flattened-matrix of the Echelle Spectra data.
 *
 * len_rows: Length of the rows of the flattened-matrix.
 *
 * len_cols: Length of the columns of the flattened-matrix.
 *
 * ------------------------------------------------------------------------------
*/
	PyArg_ParseTuple(args,"Oiidii",&marray,&len_rows,&len_cols,&precision,&Window,&option);
        if(Window%2==0){
          printf("Input error: Only uneven numbers are allowed as windows.");
          return Py_BuildValue("i",0);
         }
	 m = ARRAYD(marray);                              /* We convert our PyObject struct pointers to C-vector array.                */
         int i=0,j=0,k=0,mid_col=0;
         mid_col=getMiddleColumn(len_cols);
         double UpperTreshold,LowerTreshold;
         double* Filter = MakeVector(len_rows);          /* Vector that'll save the median filter in order to baseline substract data */
         // Output -> Freed!
	 double** Output = OrderLocation(&m[0],&Norders,len_rows,len_cols,mid_col,Window,Filter,option);
         double** Centers = MakeArray(len_cols,Norders);  /* The orders detected at the centers are the ones to be traced on the image */
         double** TraceLimits = MakeArray(Norders,2);     /* This array will save the limits of the traces.                            */
         double** CenterTracker = MakeArray(2,Norders);   /* This array will track the last successfully detected center, in order to  */
                                                          /* compare it with the detected centers by the OrderLocation() function.     */
                                                          /* The 2nd row of this array contains the ammount of pixels ommited during   */
                                                          /* the tracing, in order to stop looking for the order if a certain ammount  */
                                                          /* of pixels have been skipped successively (i.e. to detect a "trace lost"). */

         double** MCenterTracker = MakeArray(2,Norders);  /* Same as CenterTracker, but remains unchanged (so we can use it when we    */
                                                          /* track the left side of the image)                                         */

         double* Apertures = MakeVector(Norders);         /* We save the middle-calculated apertures in order to estimate deviations   */ 
                                                          /* from the calculated traces, which'll help us obtain the next pixel on the */
                                                          /* same order.                                                               */
							  
        for(i=0;i<Norders;i++){                           /* We prepare our TraceLimits array with the optimal values.                 */
	   for(j=0;j<2;j++){
	     if(j==0){
	       TraceLimits[i][0]=0;
	     }
	     else{
	       TraceLimits[i][1]=len_cols-1;
	     }
	   }
	 } 
	 
         for(j=0;j<Norders;j++){                          /* First, we track the middle column and cross-correlate all detected orders */
            Aperture=Output[j][1];                        /* by our OrderLocation() function. These orders are the ones to be traced   */
            Center=Output[j][0];                          /* by our algorithm.                                                         */
            Centers[mid_col][j]=PCrossCorrelation(&m[0],Center,Aperture,mid_col,len_cols,precision,Filter,option,len_rows);
            Apertures[j]=Aperture;
            CenterTracker[0][j]=Center;
            MCenterTracker[0][j]=Center;
         }
         FreeArray(Output,Norders);
/* Start tracing the orders to the right of the image */

         for(i=mid_col+1;i<len_cols;i++){
	    // Output -> Freed!
           // printf("Yes...\n");
            double** Output = OrderLocation(&m[0],&norders,len_rows,len_cols,i,Window,Filter,option);      /* Here, we save our first guess  */
            k=0;                                                                                    /* for the location of the centers*/
            for(j=0;j<norders;j++){
                  UpperTreshold=CenterTracker[0][k]+(Apertures[k]/2.0);                             /* Then, perform upper and lower   */
                  LowerTreshold=UpperTreshold-Apertures[k];                                         /* tresholds based on central ap.  */
                  if((Output[j][0]>=LowerTreshold) && (Output[j][0]<=UpperTreshold)){               /* If the first detected order by  */
		    Center=PCrossCorrelation(&m[0],Output[j][0],Output[j][1],i,len_cols,precision,Filter,option,len_rows);
// 		  if(k==40){
// 		      printf("Column: %d \n",i);
// 		      printf("LowerTreshold: %f, UpperTreshold: %f, Output[%d][0]: %f \n",LowerTreshold,UpperTreshold,j,Output[j][0]);
// 		      printf("Center: %f\n",Center);
// 		  }
                                                                                                    /* the OrderLocation() function is */
                    Centers[i][k]=Center;                                                           /* among the tresholds, detect it  */
                    CenterTracker[0][k]=Center;                                                     /* and perform Cross-correlation.  */
                    CenterTracker[1][k]=0;
		    //printf("Center %f, column %d, k=%d, norders=%d\n",Center, i, k, norders);
                    k++;
                  }
                  else if(Output[j][0]>=UpperTreshold){
		    /* If the requiered order couldn't be traced          */
                    CenterTracker[1][k]=CenterTracker[1][k]+1;                   /* (e.g. because of low S/N, etc.) skip to next order */
//                     if(j==norders-1){
//                       printf("The CenterTracker is: %f \n",CenterTracker[1][k]); 
//                     }
                    if(CenterTracker[1][k]==10){
		      TraceLimits[k][1]=i-10;
                      printf("Warning: Lost the trace of the %d th order at column %d.  \n",k,i);
                      printf("         The last successfully determined center was at column %d.\n",i-10);
                      CenterTracker[0][k]=-99999;                                /* Fill with a length negative enough to skip on     */
                    }                                                            /* the next detection                                */
                    k++;
		    j=0;
                  }
                  if(k==Norders)                                                 /* In the situation where there's more orders than    */
                    break;                                                       /* the ones detected at the middle column, skip to    */
            }                                                                     /* next column (all orders detected).                */
            for(l=0;l<Norders;l++){
	       if(Centers[i][l]==0){
		    /* If the requiered order couldn't be traced          */
                    CenterTracker[1][l]=CenterTracker[1][l]+1;                   /* (e.g. because of low S/N, etc.) skip to next order */
//                     if(j==norders-1){
//                       printf("The CenterTracker is: %f \n",CenterTracker[1][k]); 
//                     }
                    if(CenterTracker[1][l]==10){
		      TraceLimits[l][1]=i-10;
                      printf("Warning: Lost the trace of the %d th order at column %d.  \n",l,i);
                      printf("         The last successfully determined center was at column %d.\n",i-10);
                      CenterTracker[0][l]=-99999;                                /* Fill with a length negative enough to skip on     */
                    }                                                            /* the next detection                                */
               }
	    }
            FreeArray(Output,norders);
         }

/* Start tracing the orders to the left of the image */

         for(i=mid_col-1;i>=0;i--){
	   // Output -> Freed!
           double** Output = OrderLocation(&m[0],&norders,len_rows,len_cols,i,Window,Filter,option);/* Here, we save our first guess   */
           k=0;                                                                                     /* for the location of the centers */
           for(j=0;j<norders;j++){
                  UpperTreshold=MCenterTracker[0][k]+(Apertures[k]/2.0);                            /* Then, perform upper and lower   */
                  LowerTreshold=UpperTreshold-Apertures[k];                                         /* tresholds based on central ap.  */
                  if((Output[j][0]>=LowerTreshold) && (Output[j][0]<=UpperTreshold)){               /* If the first detected order by  */
                    Center=PCrossCorrelation(&m[0],Output[j][0],Output[j][1],i,len_cols,precision,Filter,option,len_rows); 
                                                                                                    /* the OrderLocation() function is */
                    Centers[i][k]=Center;                                                           /* among the tresholds, detect it  */
                    MCenterTracker[0][k]=Center;                                                    /* and perform Cross-correlation.  */
                    MCenterTracker[1][k]=0;
                    k++;
                  }
                  else if(Output[j][0]>=UpperTreshold){                          /* If the requiered order couldn't be traced          */
//                     if(j==norders-1){
//                       printf("The center tracker is: %f \n",MCenterTracker[1][k]);
//                     }
                    MCenterTracker[1][k]=MCenterTracker[1][k]+1;                 /* (e.g. because of low S/N, etc.) skip to next order */
                    if(MCenterTracker[1][k]==10){
		      TraceLimits[k][0]=i+10;
                      printf("Warning: Lost the trace of the %d th order at column %d.  \n",k,i);
                      printf("         The last successfully determined center was at column %d.\n",i+10);
                      MCenterTracker[0][k]=-99999;
                    }
                    k++;
		    j=0;
                  }
                  if(k==Norders)                                                 /* In the situation where there's more orders than    */
                    break;                                                       /* the ones detected at the middle column, skip to    */
           }                                                                     /* next column (all orders detected).                 */
           for(l=0;l<Norders;l++){
	       if(Centers[i][l]==0){
		    /* If the requiered order couldn't be traced          */
                    MCenterTracker[1][l]=MCenterTracker[1][l]+1;                 /* (e.g. because of low S/N, etc.) skip to next order */
                    if(MCenterTracker[1][l]==10){
		      TraceLimits[l][0]=i+10;
                      printf("Warning: Lost the trace of the %d th order at column %d.  \n",l,i);
                      printf("         The last successfully determined center was at column %d.\n",i+10);
                      MCenterTracker[0][l]=-99999;
                    }                                                        /* the next detection                                */
               }
	   }
//            printf("i=%d \n",i);
//            printf("Centers (order 0): %f \n",Centers[i][0]);
//            printf("CentersTracker (order 0): %f \n",MCenterTracker[0][0]);
           FreeArray(Output,norders);
         }
/* Start of the matrix-to-vector conversion part of the code */  

/* First we return the centers...*/

        double* theArray;
        theArray = (double*) malloc((Norders*len_cols+1)*sizeof(double));
        theArray[0]=Norders;
        for(i=0;i<Norders;i++){
           for(j=0;j<len_cols;j++){
               theArray[i*len_cols+j+1]=Centers[j][i];
            } 
        }

/* And then the trace limits...*/

        double* theArray2;
        theArray2 = (double*) malloc((Norders*2)*sizeof(double));
        for(i=0;i<Norders;i++){
           for(j=0;j<2;j++){
               theArray2[i*2+j]=TraceLimits[i][j];
            } 
        }
        
/* End of the matrix-to-vector conversion part of the code.*/


/* Finally, we create a Python "Object" List that contains the P coefficients and return it back to Python */

        PyObject *lst = PyList_New(Norders*len_cols+1);
        if (!lst)
        return NULL;
        for (i = 0; i < Norders*len_cols+1; i++) {
          PyObject *num = PyFloat_FromDouble(theArray[i]);
          if (!num) {
          Py_DECREF(lst);
          return NULL;
          }
          PyList_SET_ITEM(lst, i, num);
        }
        
        PyObject *lst2 = PyList_New(Norders*2);
        if (!lst2)
        return NULL;
        for (i = 0; i < Norders*2; i++) {
          PyObject *num = PyFloat_FromDouble(theArray2[i]);
          if (!num) {
          Py_DECREF(lst2);
          return NULL;
          }
          PyList_SET_ITEM(lst2, i, num);
        }
        
        gettimeofday(&tim, NULL);
        double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
        printf("%2.2lf seconds elapsed\n", t2-t1);
	free(theArray);
	free(theArray2);
	FreeArray(TraceLimits,Norders);
	FreeArray(Centers,len_cols);
	FreeArray(CenterTracker,2);
	FreeArray(MCenterTracker,2);
	PyObject *MyResult = Py_BuildValue("OO",lst,lst2);
        Py_DECREF(lst);
        Py_DECREF(lst2);
        return MyResult;
}

static PyObject *Order_getCenters2(PyObject *self, PyObject *args){
        struct timeval tim;
        gettimeofday(&tim, NULL);
        double t1=tim.tv_sec+(tim.tv_usec/1000000.0),precision=0,Aperture,Center;
	double *m;
	double *g;
        int len_rows,len_cols,Norders,norders,Window,option,len_guess=0;
	PyObject *marray;
	PyObject *garray;
	
/* 
 *--------------------------------THE DATA---------------------------------------
 * After initialization of the PyObject pointers, we wish to recover the following inputs:
 *
 * marray:   Vector of the flattened-matrix of the Echelle Spectra data.
 *
 * len_rows: Length of the rows of the flattened-matrix.
 *
 * len_cols: Length of the columns of the flattened-matrix.
 *
 * ------------------------------------------------------------------------------
*/
	PyArg_ParseTuple(args,"OiidiiOi",&marray,&len_rows,&len_cols,&precision,&Window,&option,&garray,&len_guess);
        if(Window%2==0){
          printf("Input error: Only uneven numbers are allowed as windows.");
          return Py_BuildValue("i",0);
         }
	 m = ARRAYD(marray);                              /* We convert our PyObject struct pointers to C-vector array.                */
         g = ARRAYD(garray);
	 int i=0,j=0,k=0,mid_col=0;
         mid_col=getMiddleColumn(len_cols);
         double UpperTreshold,LowerTreshold;
         double* Filter = MakeVector(len_rows);          /* Vector that'll save the median filter in order to baseline substract data */
         // Output -> Freed!
	 double** Output = OrderLocation(&m[0],&Norders,len_rows,len_cols,mid_col,Window,Filter,option);
	 double** Output2 = MakeArray(len_guess,2);
	 for(i=0;i<len_guess;i++){
	   for(j=0;j<2;j++){
	     Output2[i][j]=g[i*2+j];
	   }
	 }
         double* Apertures = MakeVector(len_guess);
         double** Centers = MakeArray(len_cols,len_guess);  /* The orders detected at the centers are the ones to be traced on the image */
         double** TraceLimits = MakeArray(len_guess,2);     /* This array will save the limits of the traces.                            */
 	 double** CenterTracker = MakeArray(2,len_guess);   /* This array will track the last successfully detected center, in order to  */
                                                          /* compare it with the detected centers by the OrderLocation() function.     */
                                                          /* The 2nd row of this array contains the ammount of pixels ommited during   */
                                                          /* the tracing, in order to stop looking for the order if a certain ammount  */
                                                          /* of pixels have been skipped successively (i.e. to detect a "trace lost"). */
         double** MCenterTracker = MakeArray(2,len_guess);  /* Same as CenterTracker, but remains unchanged (so we can use it when we    */
                                                          /* track the left side of the image)                                         */
         //double* Apertures = MakeVector(len_guess);         /* We save the middle-calculated apertures in order to estimate deviations  */ 
                                                          /* from the calculated traces, which'll help us obtain the next pixel on the */
                                                          /* same order.                                                               */
         for(j=0;j<len_guess;j++){                          /* First, we track the middle column and cross-correlate all detected orders */
            Aperture=Output2[j][1];                        /* by our OrderLocation() function. These orders are the ones to be traced   */
            Center=Output2[j][0];                          /* by our algorithm.                                                         */
            Centers[mid_col][j]=PCrossCorrelation(&m[0],Center,Aperture,mid_col,len_cols,precision,Filter,option,len_rows);
            Apertures[j]=Aperture;
            CenterTracker[0][j]=Center;
            MCenterTracker[0][j]=Center;
         }
         FreeArray(Output,Norders);
	 FreeArray(Output2,len_guess);
	 Norders=len_guess;
	 
	 for(i=0;i<Norders;i++){                           /* We prepare our TraceLimits array with the optimal values.                 */
	   for(j=0;j<2;j++){
	     if(j==0){
	       TraceLimits[i][0]=0;
	     }
	     else{
	       TraceLimits[i][1]=len_cols-1;
	     }
	   }
	 } 

/* Start tracing the orders to the right of the image */
         for(i=mid_col+1;i<len_cols;i++){
	    // Output -> Freed!
            double** Output = OrderLocation(&m[0],&norders,len_rows,len_cols,i,Window,Filter,option);      /* Here, we save our first guess  */
            k=0;                                                                                    /* for the location of the centers*/
            for(j=0;j<norders;j++){
                  UpperTreshold=CenterTracker[0][k]+(Apertures[k]/2.0);                             /* Then, perform upper and lower   */
                  LowerTreshold=UpperTreshold-Apertures[k];                                         /* tresholds based on central ap.  */
                  if((Output[j][0]>=LowerTreshold) && (Output[j][0]<=UpperTreshold)){               /* If the first detected order by  */
		    Center=PCrossCorrelation(&m[0],Output[j][0],Output[j][1],i,len_cols,precision,Filter,option,len_rows);
// 		  if(k==40){
// 		      printf("Column: %d \n",i);
// 		      printf("LowerTreshold: %f, UpperTreshold: %f, Output[%d][0]: %f \n",LowerTreshold,UpperTreshold,j,Output[j][0]);
// 		      printf("Center: %f\n",Center);
// 		  }
                                                                                                    /* the OrderLocation() function is */
                    Centers[i][k]=Center;                                                           /* among the tresholds, detect it  */
                    CenterTracker[0][k]=Center;                                                     /* and perform Cross-correlation.  */
                    CenterTracker[1][k]=0;
                    k++;
                  }
                  else if(Output[j][0]>=UpperTreshold){
		    /* If the requiered order couldn't be traced          */
                    CenterTracker[1][k]=CenterTracker[1][k]+1;                   /* (e.g. because of low S/N, etc.) skip to next order */
                    if(CenterTracker[1][k]==10){
      		      TraceLimits[k][1]=i-10;
                      printf("Warning: Lost the trace of the %d th order at column %d.  \n",k,i);
                      printf("         The last successfully determined center was at column %d.\n",i-10);
                      CenterTracker[0][k]=-99999;                                /* Fill with a length negative enough to skip on     */
                    }                                                            /* the next detection                                */
                    k++;
		    j=0;
                  }
                  if(k==Norders)                                                 /* In the situation where there's more orders than    */
                    break;                                                       /* the ones detected at the middle column, skip to    */
            }                                                                     /* next column (all orders detected).                */
            FreeArray(Output,norders);
         }

/* Start tracing the orders to the left of the image */
         for(i=mid_col-1;i>=0;i--){
	   // Output -> Freed!
           double** Output = OrderLocation(&m[0],&norders,len_rows,len_cols,i,Window,Filter,option);/* Here, we save our first guess   */
           k=0;                                                                                     /* for the location of the centers */
           for(j=0;j<norders;j++){
                  UpperTreshold=MCenterTracker[0][k]+(Apertures[k]/2.0);                            /* Then, perform upper and lower   */
                  LowerTreshold=UpperTreshold-Apertures[k];                                         /* tresholds based on central ap.  */
                  if((Output[j][0]>=LowerTreshold) && (Output[j][0]<=UpperTreshold)){               /* If the first detected order by  */
                    Center=PCrossCorrelation(&m[0],Output[j][0],Output[j][1],i,len_cols,precision,Filter,option,len_rows); 
                                                                                                    /* the OrderLocation() function is */
                    Centers[i][k]=Center;                                                           /* among the tresholds, detect it  */
                    MCenterTracker[0][k]=Center;                                                    /* and perform Cross-correlation.  */
                    MCenterTracker[1][k]=0;
                    k++;
                  }
                  else if(Output[j][0]>=UpperTreshold){                          /* If the requiered order couldn't be traced          */
                    MCenterTracker[1][k]=MCenterTracker[1][k]+1;                 /* (e.g. because of low S/N, etc.) skip to next order */
                    if(MCenterTracker[1][k]==10){
      		      TraceLimits[k][1]=i+10;
                      printf("Warning: Lost the trace of the %d th order at column %d.  \n",k,i);
                      printf("         The last successfully determined center was at column %d.\n",i+10);
                      MCenterTracker[0][k]=-99999;
                    }
                    k++;
		    j=0;
                  }
                  if(k==Norders)                                                 /* In the situation where there's more orders than    */
                    break;                                                       /* the ones detected at the middle column, skip to    */
           }                                                                     /* next column (all orders detected).                 */
//            printf("i=%d \n",i);
//            printf("Centers (order 0): %f \n",Centers[i][0]);
//            printf("CentersTracker (order 0): %f \n",MCenterTracker[0][0]);
           FreeArray(Output,norders);
         }

/* Start of the matrix-to-vector conversion part of the code */    
        double* theArray;
        theArray = (double*) malloc((Norders*len_cols+1)*sizeof(double));
        theArray[0]=Norders;
        for(i=0;i<Norders;i++){
           for(j=0;j<len_cols;j++){
               theArray[i*len_cols+j+1]=Centers[j][i];
            } 
        }
        
        double* theArray2;
        theArray2 = (double*) malloc((Norders*2)*sizeof(double));
        for(i=0;i<Norders;i++){
           for(j=0;j<2;j++){
               theArray2[i*2+j]=TraceLimits[i][j];
            } 
        }
        
/* End of the matrix-to-vector conversion part of the code.*/


/* Finally, we create a Python "Object" List that contains the P coefficients and return it back to Python */

        PyObject *lst = PyList_New(Norders*len_cols+1);
        if (!lst)
        return NULL;
        for (i = 0; i < Norders*len_cols+1; i++) {
          PyObject *num = PyFloat_FromDouble(theArray[i]);
          if (!num) {
          Py_DECREF(lst);
          return NULL;
          }
          PyList_SET_ITEM(lst, i, num);
        }
        
        PyObject *lst2 = PyList_New(Norders*2);
        if (!lst2)
        return NULL;
        for (i = 0; i < Norders*2; i++) {
          PyObject *num = PyFloat_FromDouble(theArray2[i]);
          if (!num) {
          Py_DECREF(lst2);
          return NULL;
          }
          PyList_SET_ITEM(lst2, i, num);
        }
        gettimeofday(&tim, NULL);
        double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
        printf("%2.2lf seconds elapsed\n", t2-t1);
	free(theArray);
	free(theArray2);
        free(Filter);
        free(Apertures);
	FreeArray(TraceLimits,Norders);
	FreeArray(Centers,len_cols);
	FreeArray(CenterTracker,2);
	FreeArray(MCenterTracker,2);
	PyObject *MyResult = Py_BuildValue("OO",lst,lst2);
        Py_DECREF(lst);
        Py_DECREF(lst2);
        return MyResult;
}

static PyMethodDef OrderMethods[] = {
        {"CCFCenter",Order_CCFCenter, METH_VARARGS, "Given a vector and a precision, finds the center via CCF with a gaussian."},
	{"Locate", Order_Locate, METH_VARARGS, "Locates orders on the image using threshold-detection methods and the Hough Transform."},
	{"getCenters", Order_getCenters,METH_VARARGS, "This method detects the centers of each order using the location algorithm as a first approximation to the centers using cross-correlation with gaussians"},
	{"getCenters2", Order_getCenters2,METH_VARARGS, "This method detects the centers of each order given a first guess for the centers and apertures."},
	{NULL, NULL, 0, NULL}
};

void initOrder(void){
	(void) Py_InitModule("Order", OrderMethods);
}

/*-------------------------------------------------------------  
* OrderLocation: A FUNCTION FOR THE LOCATION OF ECHELLE ORDERS
* -------------------------------------------------------------
*
* This function locates the order in a given column by two methods, depending on the input of mid_col. If mid_col is
* specified, this algorithm detects the orders present in the given column (it doesn't have to be the middle column, the name
* was just a reference). In the case where mid_col isn't specified (less than zero), the algorithm calculates the orders in the
* arithmetical middle column.
*
*/


double** OrderLocation(double *m,int *norders,int len_rows,int len_cols,int mid_col,int Window,double* Filter,int option){
        double* M = MakeVector(len_rows);        /* Create a matrix array for our image.                                  */
        double* B = MakeVector(len_rows);        /* Array that will save measurements correspondig to order detections    */
        double PartialMAD=0,Mean=0,Sigma=0,Threshold=0;
        int i,s=1,counter,CurrentMax=0;
        if(mid_col<0){
           mid_col=getMiddleColumn(len_cols);
          }
        for(i=0;i<len_rows;i++){                          /* Fill our matrix with the image data                                  */
	          M[i]=m[i*len_cols+mid_col];
        }
        //printf("Column: %d \n",mid_col);

/* Up to here, the M matrix contains the middle pixel prepared for our measurements. Now we perform the order detection:        */

	  /* Start of baseline substraction... */
            double* MedianBackground = MedianFilter(M,len_rows,Window);
            for(i=0;i<len_rows;i++){
               Filter[i]=MedianBackground[i];
               M[i]=M[i]-MedianBackground[i];
            }
            free(MedianBackground);
        // printf("Filter done...\n");
    	  /* End of baseline substraction */
	  
	  /* Now we obtain a median background from our result in order to obtain
	     a sample and obtain it's statistical properties so we can "create" our
	     threshold. */
	  
	    double* BGSample = MedianFilter(M,len_rows,Window);
	   // printf("Double filter done...\n");
	    Mean=MeanCalculator(BGSample,0,len_rows-1);
            PartialMAD=MAD(BGSample,0,len_rows-1,Mean);
            Sigma=PartialMAD*1.25331414;
            Threshold=Mean+5*Sigma;
            free(BGSample);
           // printf("Statistics done...\n");
	  /* End of the threshold finding algorithm */
	    
            for(i=0;i<len_rows-1;i++){
	       // printf("row: %d of %d\n",i,len_rows);
               if(M[i]!=-9999){
                   if(M[i+1]>Threshold){                      /* ...check whether the next value is above the 3-sigma...           */
                                                                /* ... and check if there are any "orders" on the
                                                                       following pixels. If the next pixels have detections...       */
                     if(CheckOrder(M,Threshold,i,len_rows)==1){
                        CurrentMax=CheckMax(M,i,len_rows);
                        for(s=1;CheckEndOrder(M,i,s,Threshold,CurrentMax,len_rows)==0;s++){
			    if(i+s<len_rows)
                               B[i+s]=M[i+s];                /*...we save the values: 0 represent no detections. This is
                                                                 the Hough Transform of our measurement.                             */
                            else
			       break;			  
			}
//                         printf("i+s: %d \n",i+s);
                        i=i+s;                                  /* And we "jump" to the first pixel after the order                  */
                     }
                     else{
                     M[i+1]=-9999;                           /* If it wasn't an order, then the next deviation is ommited on 
                                                                   the MAD sum (detected as cosmic ray or bad pixel).                */
                     }
                  }
               }
            }
            // printf("Detections done...\n");
/* We separated the detections. Now proceed to evaluate our detection counts using the Hough Transform */

//         printf("B[0][4]=%f\n",B[0][4]);
//         printf("B[0][5]=%f\n",B[0][5]);
//         printf("B[0][6]=%f\n",B[0][6]);
        int orders=0,aperture=0;

        for(i=0;i<len_rows-3;i++){
            counter=0;
            if(aperture==0){
              if(B[i]>0 && B[i+1]>0 && B[i+2]>0){           /* In other words, we want that the detection happens on the      */
                 counter++;                                 /* following 3 rows (if not, the signal was too low to calculate  */
                                                            /* a proper aperture and therefore won't count)                  */
              }
            }
            else{
              if(B[i]>0){ 
                 counter++;
              }
            }
            if(counter==1 && aperture==0){
              orders++;
              aperture=1;
            }
            else if(counter==1 && aperture!=0){
              aperture++;
            }
            else if(counter==0 && aperture!=0){
              aperture=0;
            }
        }

        // printf("Finding orders in the image... \n");
        // printf("%d orders found.\n",orders);
        double** Output = MakeArray(orders,2);
        double sum=0,dummysum=0;
        *norders=orders;
        int index=0;
        orders=0;
        aperture=0;
        counter=0;
        for(i=0;i<len_rows-3;i++){
            counter=0;
            sum=0;
            if(aperture==0){
              if(B[i]>0 && B[i+1]>0 && B[i+2]>0){           /* In other words, we want that the detection happens on the      */
                 counter++;                                 /* following 3 rows (if not, the signal was too low to calculate  */
                 sum=B[i];                                  /* a proper aperture and therefore won't counte)                  */
              }
            }
            else{
              if(B[i]>0){ 
                 counter++;
                 sum=B[i];
              }
            }
            if(counter==1 && aperture==0){
              orders++;
              aperture=1;
              dummysum=sum;
            }
            else if(counter==1 && aperture!=0){
              aperture++;
              if(sum>=dummysum){
                index=i;
                dummysum=sum;
              }
            }
            else if(counter==0 && aperture!=0){
              Output[orders-1][0]=index;
              Output[orders-1][1]=aperture;
              aperture=0;
            }
        }
        free(M);
        free(B);
	if(option==2){
	  double** Output2 = FitDetection(Output,orders,len_rows,norders);
	  return Output2;
	}
        return Output;
}

int getMiddleColumn(int len_cols){
          int mid_col;
          if(len_cols%2==0){                                    /* Check whether len_cols is even or odd in order to find the   */
           return mid_col=len_cols/2;                           /* middle column pixel.                                         */
          }
          else{
           return mid_col=((len_cols-1)/2)+1;
          }
}

/*
 *  This Quickselect routine is based on the algorithm described in
 *  "Numerical recipes in C", Second Edition,
 *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
 *  This code by Nicolas Devillard - 1998. Public domain.
 */

double quick_select(double arr[], int n){
    int low, high ;
    int median;
    int middle, ll, hh;

    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
        if (high <= low) /* One element only */
            return arr[median] ;

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]) ;
            return arr[median] ;
        }

    /* Find median of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

    /* Swap low item (now in position middle) into position (low+1) */
    ELEM_SWAP(arr[middle], arr[low+1]) ;

    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
        do ll++; while (arr[low] > arr[ll]) ;
        do hh--; while (arr[hh]  > arr[low]) ;

        if (hh < ll)
        break;

        ELEM_SWAP(arr[ll], arr[hh]) ;
    }

    /* Swap middle item (in position low) back into correct position */
    ELEM_SWAP(arr[low], arr[hh]) ;

    /* Re-set active partition */
    if (hh <= median)
        low = ll;
        if (hh >= median)
        high = hh - 1;
    }
}

// #undef ELEM_SWAP

double* MedianFilter(double* A,int len_A,int Window){
    int i;
    double* M = MakeVector(len_A);
    for(i=0;i<len_A;i++){
       M[i]=MedianWindowing(A,i,len_A,Window);
    }
    return M;
}

double MedianWindowing(double* A,int pos,int len_A,int Window){
    int i,cp,lp;
    lp=pos-((Window-1)/2);
    double Result=0;
    double* V = MakeVector(Window);
    for(i=0;i<Window;i++){
       cp=lp+i;
       if(cp<0){
         V[i]=A[0];
       }
       else if(cp>=len_A){
         V[i]=A[len_A-1];
       }
       else{
         V[i]=A[cp];
       }
    }
    Result=quick_select(V,Window);
    free(V);
    return Result;
}

double** MakeArray(int rows, int columns){
    int i,j;
    double** theArray;
    theArray = (double**) malloc(rows*sizeof(double*));
    for(i=0;i<rows;i++)
        theArray[i] = (double*) malloc(columns*sizeof(double));

    /* Fill the array with zeroes (i.e. we clean it) */

    for(i=0;i<rows;i++){
       for(j=0;j<columns;j++){
         theArray[i][j]=0.0;
       }
    }

    return theArray;
}

double* MakeVector(int nelements){
    double* Vector;
    int j;
    Vector = (double*) malloc(nelements*sizeof(double));

    for(j=0;j<nelements;j++){
         Vector[j]=0.0;
    }
     return Vector;
}

void FreeArray(double** theArray,int rows){
    int i;
    for(i=0;i<rows;i++){
       free(theArray[i]);
    }
    free(theArray);
}

int CheckOrder(double* M,double Threshold,int i,int len_rows){
    int OrderFound=0;
    if(M[i+1]>=Threshold && i+3<len_rows){
       if(M[i+2]>=Threshold && M[i+3]>=Threshold) 
          OrderFound=1; /* If the 3 contiguous rows have detections, we found an order! */
    }
    return OrderFound;
}

double MeanCalculator(double* M,int imin,int i){
    int n=0,k;
    for(k=imin;k<=i;k++){
       if(M[k]!=-9999){
         n++;
       }
    }
//     double* S = MakeVector(n);
    double sum=0;
    n=0;
    for(k=imin;k<=i;k++){
       if(M[k]!=-9999){
//          S[n]=M[k][j];
         sum=M[k]+sum;
         n++;
       }
    }
//     return quick_select(S,n);
    return sum/((double)n);
}

double MAD(double* M,int imin,int i,double Mean){
    int n=0,k;
    for(k=imin;k<=i;k++){
       if(M[k]!=-9999){
         n++;
       }
    }
    double sum=0;
    n=0;
    for(k=imin;k<=i;k++){
       if(M[k]!=-9999){
         sum=fabs(Mean-M[k])+sum;
         n++;
       }
    }
    return sum/((double)n);
}

int CheckMax(double* M,int i,int len_rows){
    int k;
    double DummyMax=0;
    for(k=0;k>=0;k++){
       if(i+k+1<len_rows){
         DummyMax=M[i+k];
       }
       else{
         break;
       }
       if(i+k!=0){
         if((M[i+k+1]<=DummyMax && M[i+k-1]<=DummyMax) || (M[i+k-1]>=DummyMax && M[i+k+1]<=DummyMax))
            break;
       }
    }
    return i+k;
}

int CheckEndOrder(double* M,int i,int s,double Threshold,int Max,int len_rows){
    int StepFound=0;
    if(Max<i+s){
      if(i+s<len_rows){
        if(M[i+s]<Threshold)   /* If we get back to our "count floor", then the order has ended */
          StepFound=1;
      }
    }

    return StepFound;
}

/* PolynomialFit.

   This function returns the coefficients An,An-1...A0 of a polynomial fit of the form:

      p(x)=An*x^n+...A2*x^2+A1*x+A0

   To the data using chi-square minimisation.

*/

double* PolynomialFit(double* data_x,double* data_y,int n,int order){
    double** A = MakeArray(order+1,order+1);
    double* X = MakeVector(order*2+1);
    double* b = MakeVector(order+1);
    double* x = MakeVector(order+1);
    double sum;
    double sum2;
    int i,j;
    for(i=0;i<order*2+1;i++){
       sum=0;
       sum2=0;
       for(j=0;j<n;j++){
          sum=pow(data_x[j],i)+sum;                        /* We save the elements of the A matrix */
       }
       if(i<order+1){
         for(j=0;j<n;j++){
            sum2=pow(data_x[j],i)*data_y[j]+sum2;               /* We save the elements of the b vector */
         }
         b[i]=sum2;
       }
       X[i]=sum;
    }
    /* Now we fill our A matrix */
    for(i=0;i<order+1;i++){
       for(j=0;j<order+1;j++){
          A[i][j]=X[order-j+i];                       /* We fill the (diagonal) matrix       */
       }
    }

    LinearSolver(A,b,x,order+1);
    FreeArray(A,order+1);
    free(X);
    free(b);
    return x;
}

/* PolynomialEval.

   This function returns the p(x) value if the coefficients An,An-1...A0 of a polynomial fit
   of the form:

      p(x)=An*x^n+...A2*x^2+A1*x+A0

   are given along with the x values to evaluate.

*/

double* PolynomialEval(double* A,double* x,int order,int len_x){
  double* p = MakeVector(len_x);
  int i,j;
  double sum=0;
  for(i=0;i<len_x;i++){
    sum=0;
    for(j=0;j<order+1;j++){
      sum=sum+A[order-j]*pow(x[i],j);
    }
    p[i]=sum;
  }
  return p;
}

void LinearSolver(double** A,double* b,double* x,int length){
       int i,j;

       gsl_matrix *AGSL = gsl_matrix_alloc(length,length);   /* We allocate memory for our GSL matrices and vectors    */
       gsl_vector *xGSL = gsl_vector_alloc(length);
       gsl_vector *bGSL = gsl_vector_alloc(length);

       for(i=0;i<length;i++){
          for(j=0;j<length;j++){
             gsl_matrix_set(AGSL,i,j,A[i][j]);               /* Set the obtained values of the C_qp matrix to "A"      */
          }
          gsl_vector_set(bGSL,i,b[i]);                       /* Set the obtained values of the X_q vector to "b"       */
       } 
       gsl_vector *tau = gsl_vector_alloc(length);
       gsl_linalg_QR_decomp(AGSL,tau);
       gsl_linalg_QR_solve(AGSL,tau,bGSL,xGSL);
       for(i=0;i<length;i++){
          x[i]=gsl_vector_get(xGSL,i);                       /* Set the solution in each B_q vector term               */
       } 
}

double PCrossCorrelation(double *m,double Loc,double Ap,int CurrentColumn,int len_cols,double precision,double* Filter,int option,int len_rows){
       int printcolumn = -1;
       int Location = (int)(Loc);
       int Aperture = (int)(Ap);
       int CentralAperture=(int)((Ap/2.0)+1.5);  /* We are extra-cautious and add 1 extra pixel to the estimated apertures  */
       if(option==0){
        /* Search centers via cross-correlation */ 
        int i,j,n,dummy,len_g=2*CentralAperture;           /* len_g is the length of the window (in pixels) to be cross-correlated with a gaussian */
	if(Location-CentralAperture<0){
          CentralAperture = Location;
          len_g=2*CentralAperture;
	}
        if(Location+CentralAperture>=len_rows){
          len_g=CentralAperture-Location+len_rows-1;
        }
        double CurrentValue,CurrentValue2,comparison;
        n=(int)(((double)len_g)/precision);   
        double* oversampled_data = MakeVector(n);
        int spacing=(int)(1.0/precision);
        dummy = Location-CentralAperture;
        if(printcolumn == CurrentColumn){
         printf(" \n\n Column %d \n \n",CurrentColumn);
         printf("Location: %d, CentralAperture: %d, len_g: %d \n",Location,CentralAperture,len_g);
         printf("spacing: %d, \n \n",spacing);
         printf("data: ");
        }
        //double sump = m[dummy*len_cols+CurrentColumn]-Filter[dummy];  
        double sump = 0;
        for(i=0;i<len_g;i++){
           dummy=Location-CentralAperture+i;
           /* dummy is the current row we are working at. Recall that m is the 
              flattened matrix, so dummy*len_cols leaves us at the current row,
              while summing by CurrentColumn leaves us at the current pixel 
              of the image we are working at. Filter, on the other hand, is 
              a vector because it is the (median) filter in a given column.

              We multiply by "precision" because it is the fraction of the signal in the pixel.
           */
           CurrentValue=m[dummy*len_cols+CurrentColumn]; /* We baseline substract AND get fraction of signal*/
           CurrentValue2=m[(dummy+1)*len_cols+CurrentColumn]; /* Same for the next pixel. Why? Because... */
           //CurrentValue=(m[dummy*len_cols+CurrentColumn]-Filter[dummy]); /* We baseline substract AND get fraction of signal*/
           //CurrentValue2=(m[(dummy+1)*len_cols+CurrentColumn]-Filter[dummy+1]); /* Same for the next pixel. Why? Because... */
           if(printcolumn == CurrentColumn){
              printf("%f,",m[dummy*len_cols+CurrentColumn]);
           }
           sump = sump+CurrentValue;
           for(j=0;j<spacing;j++){
              /* We are comparing different "strips" in pixel space. We divide each pixel in spacing = 1/precision strips. Therefore, we 
                 have to check in which pixel we are, considering that with a,b integers, the pixel indicator (a,b) lies just at the center 
                 of the pixel. Therefore, the boundary of the pixel is at (a+-0.5,b)...*/
              comparison=(1.0/(double)spacing)*(double)j; /* This variable measures where we are in fractions of pixels */
              /* The g vector saves the signal values at each of these strips */
              if(comparison<0.5){
              oversampled_data[i*spacing+j]=CurrentValue;                 /* Note that by this definition, our g vector is product between a         */
                                                           /* rectangle function and m (see Bracewell, 2000, 3d Ed. Chapter 4, pp. 55)*/
              }
              else if(comparison>0.5){
              oversampled_data[i*spacing+j]=CurrentValue2;                 /* Note that by this definition, our g vector is product between a         */
                                                            /* rectangle function and m (see Bracewell, 2000, 3d Ed. Chapter 4, pp. 55)*/
              }
              else if(comparison==0.5){
              oversampled_data[i*spacing+j]=(CurrentValue+CurrentValue2)/2.0;
              }
           }                                               
        }
        double* template = MakeVector(n);
        double exponent=0;                                 /* Proceed to create our gaussian template..                               */
        double template_mean_window = ((double)len_g/2.0);            /* Mean of the gaussian template w/r to the len_g pixel window */
        double sigma=1.0;
        if(printcolumn == CurrentColumn){
         printf("\n\nsump: %f \ntemp_mean: %f, \nA: %f\nn: %d\n\n template:",sump,template_mean_window,sump*NormA/sigma,n);
        }
        for(i=0;i<n;i++){
           exponent=-pow((-(template_mean_window)+((double)i)*precision)/sigma,2.0)/2.0;  /* Gaussian function of 3*sigma = 1.5 pix, */
           template[i]=sump*NormA*exp(exponent)/sigma;                                                 /* "covering" 3 pixels (normalized).       */
           if(printcolumn == CurrentColumn){
             printf("%f,",template[i]);
           }
        }
        if(printcolumn == CurrentColumn){
           printf("\n\n");
        }
        double* c = MakeVector(n);
        double sum=0;
        for(i=0;i<n;i++){
           sum=0;
           for(j=0;j<n;j++){
              if(j-i<n && j-i>=0){
                sum=oversampled_data[j]*template[j-i]+sum;
              }
           }
           c[i]=sum;
        }
        double* max = WhereMax(c,n); // Return the maximum shift and value of the function at that max shift
        double ccf_shift_right = max[0]*precision;
        double ccf_right_value = max[1];
        sum=0;
        for(i=0;i<n;i++){
           sum=0;
           for(j=0;j<n;j++){
              if(j-i<n && i-j>=0){
                sum=oversampled_data[j]*template[i-j]+sum;
              }
           }
           c[i]=sum;
        }
        double* max2 = WhereMax(c,n);
        double ccf_shift_left = max2[0]*precision;
        double ccf_left_value = max2[1];
        free(max);
        free(max2);
        free(c);
        free(oversampled_data);
        free(template);
        if(CurrentColumn == printcolumn){
          printf("ccf_left - temp: %f \nccf_r+temp: %f \n",ccf_shift_left-template_mean_window,template_mean_window+ccf_shift_right);
          printf("ccf_left: %f \nccf_right: %f \n",ccf_shift_left,ccf_shift_right);
          printf("possible return values: %f, %f \n",ccf_shift_left-template_mean_window+(double)Location - (double)CentralAperture,template_mean_window+ccf_shift_right+(double)Location - (double)CentralAperture);
        }
        /* We return the shift given by the 'left' ccf shift only if it doesn't go around again to the 
           current location and only if the ccf_shift_right is zero (i.e., when the real center is 
           different from the current location */
        if(ccf_left_value>ccf_right_value){
          if(CurrentColumn == printcolumn){
            printf("returned value: %f \n",ccf_shift_left-template_mean_window+(double)Location - (double)CentralAperture);
          }
          return ccf_shift_left-template_mean_window+(double)Location - (double)CentralAperture;
        }
        if(CurrentColumn == printcolumn){
          printf("returned value: %f \n",template_mean_window+ccf_shift_right+(double)Location - (double)CentralAperture);
        }
        return template_mean_window+ccf_shift_right+(double)Location - (double)CentralAperture;
       }
       else if(option==1 || option==2){
         /* Else, search for centers via centroid method */
	 int i,beginning=(int)(Location-CentralAperture),end=(int)(Location+CentralAperture);
	 double sumX=0,sumXY=0,dummyv=0;
	 if(beginning<0){
	   beginning=0;
	 }
	 if(end>=len_rows){
	   end=len_rows-1;
	 }
	 for(i=beginning;i<=end;i++){
	    dummyv=m[i*len_cols+CurrentColumn]-Filter[i];
	    sumXY=dummyv*(double)i+sumXY;
	    sumX=dummyv+sumX;
	 }
	 return sumXY/sumX;
       }
       return -1;
}

double* WhereMax(double* Vector, int n){
       double* OutputVector = MakeVector(2);
       int i,N=1;
       int indexmax=0;
       double currentmax=Vector[0];
       for(i=1;i<n;i++){
           if(Vector[i]>currentmax){ 
              currentmax=Vector[i];
              indexmax=i;
              N=1;
           }
           else if(Vector[i]<currentmax){
           }
           else if(Vector[i]==currentmax && Vector[i]==Vector[i-1]){
              N++;
              indexmax++;
           }
       }
//        printf("Found max in CCF at index %f \n",(double)indexmax/(double)N);
       OutputVector[0] = (double)indexmax/(double)N;
       OutputVector[1] = currentmax;
       return OutputVector;
}

double** FitDetection(double** A,int len_rows,int len_rows_image,int *norders){
  int i,j,option=1,order=3,initial_rows,s_L=0,s_U=0;
  double LastOne=(double)len_rows_image;
  double* new_rows = MakeVector(3); // FREED
  double* DetectedOC = MakeVector(len_rows); // FREED
  double* Y = MakeVector(len_rows); // FREED
  double* Order = MakeVector(len_rows); // FREED
  double* S = MakeVector(order+1); // FREED
  initial_rows=len_rows;
  for(i=0;i<len_rows;i++){
    DetectedOC[i]=A[i][0];
    Order[i]=i;
  }
  while(option==1){
    S=PolynomialFit(Order,DetectedOC,len_rows,order);
    Y=PolynomialEval(S,Order,order,len_rows);
    for(i=0;i<len_rows;i++){
    }
    SigmaClipping(new_rows,Order,DetectedOC,Y,len_rows);
    if(new_rows[0]==len_rows-1){
      Order = DeleteElement(Order,new_rows[1],len_rows);
      DetectedOC = DeleteElement(DetectedOC,new_rows[1],len_rows);
      Y = DeleteElement(Y,new_rows[1],len_rows);
      len_rows=new_rows[0];
    }
    else{
      option=0;
    }
  }

  /* First we check where the evaluation of the fitted polynomial gives us a
     value lower than 0 */
  
  double* DummyVector = MakeVector(1);
  while(option==0){
    s_L--;
    DummyVector[0]=s_L;
    DummyVector = PolynomialEval(S,DummyVector,order,1);
    if(DummyVector[0]<0 || DummyVector[0]>LastOne){
      option=1;
      s_L++;
    }
    LastOne=DummyVector[0];
  }

  /* Then we check where the evaluation of the fitted polynomial gives us a
     value higher than len_rows */
  
  LastOne=0;
  while(option==1){
    s_U++;
    DummyVector[0]=s_U;
    DummyVector = PolynomialEval(S,DummyVector,order,1);
    if(DummyVector[0]>=len_rows_image || LastOne>DummyVector[0]){
      option=0;
      s_U--;
    }
    LastOne=DummyVector[0];
  }
  
  /* Now we extrapolate the orders on the image... */
  
  int length = abs(s_U-s_L)+1;
  *norders=length;
  double** Output = MakeArray(length,2);
  for(i=0;i<length;i++){
    DummyVector[0]=s_L+i;
    DummyVector =  PolynomialEval(S,DummyVector,order,1);
    Output[i][0] = DummyVector[0];
    for(j=0;j<initial_rows;j++){
      if(abs(A[j][0]-Output[i][0])<3*new_rows[2]){
	Output[i][1]=A[j][1]+6*new_rows[2]; // If we find the aperture, we fill it.
	j=initial_rows+1;
	break;
      }
    }
    if(j!=initial_rows+1){
      Output[i][1]=3+10*new_rows[2]; // If not, we make it equal to minimum aperture+error
    }
  }
  FreeArray(A,initial_rows);
  free(DummyVector);
  free(new_rows);
  free(DetectedOC);
  free(Y);
  free(Order); 
  free(S);
  return Output;
}

double* DeleteElement(double* A,int Element,int rows){
  int i,swtch=0;
  double* NewVector = MakeVector(rows-1);
  for(i=0;i<rows;i++){
    if(i!=Element){
       NewVector[i-swtch]=A[i];
    }
    else
      swtch=1;
  }
  free(A);
  return NewVector;
}

void SigmaClipping(double* vector,double* x,double* Data,double* Model,int rows){
  int i;
  double Sigma = getSigma(Data,Model,rows),distance=0,c_distance;
  for(i=0;i<rows;i++){
    c_distance=Data[i]-Model[i];
    if(c_distance<0)
      c_distance=-c_distance;
    if(c_distance>3*Sigma){
      if(c_distance>=distance){
        vector[1]=i;
	distance=c_distance;
      }
    }
  }
  vector[2]=Sigma;
  if(distance!=0){
     vector[0]=rows-1;
  }
}

double getSigma(double* Data,double* Model,int rows){
  int i;
  double sum=0.0,value;
  for(i=0;i<rows;i++){
    value=Data[i]-Model[i];
    if(value>=0)
       sum=value+sum;
    else
       sum=-value+sum;
  }
  return ((sum)/((double)(rows)))*1.4826;
}

double CCrossCorrelation(double *m,int len,double precision,double sigma){
       int ntemp,i,j;
       double x,mutemp,Atemp=0,comparison;
       for(i=0;i<len;i++){
           Atemp = m[i] + Atemp;
       }
       Atemp = Atemp*NormA/sigma;
       ntemp = (int)((double)(len)/precision);
       double* xtemp = MakeVector(ntemp);
       double* ytemp = MakeVector(ntemp);
       double* yupsampled = MakeVector(ntemp);
       mutemp = (double)(len)/2.0;
       for(i=0;i<ntemp;i++){
          x = i*precision;
          xtemp[i] = x;
          ytemp[i] = Atemp*exp(-pow(x-mutemp,2.0)/(2.*(pow(sigma,2.0))));
       }
       int spacing = (int)(1./precision);
       for(i=0;i<len-1;i++){
          for(j=0;j<spacing;j++){
              comparison = (1./(double)(spacing))*j;
              if(comparison < 0.5){
                 yupsampled[i*spacing+j] = m[i];
              }
              else if(comparison > 0.5){
                 yupsampled[i*spacing+j] = m[i+1];
              }
              else{
                 yupsampled[i*spacing+j] = (m[i+1]+m[i])/2.0;
              }
          } 
       }
       double* ccf = MakeVector(ntemp);
       double tsum = 0.;
       for(i=0;i<ntemp;i++){
           tsum = 0.;
           for(j=0;j<ntemp;j++){
               if(j-i<ntemp && j-i>=0){
                  tsum = yupsampled[j]*ytemp[j-i]+tsum;
               }
           }
           ccf[i] = tsum;
       }
       double* max = WhereMax(ccf, ntemp);
       double ccf_max_right = max[1];
       double ccf_shift_right = max[0]*precision;
       for(i=0;i<ntemp;i++){
           tsum = 0.;
           for(j=0;j<ntemp;j++){
               if(j-i<ntemp && i-j>=0){
                  tsum = yupsampled[j]*ytemp[i-j]+tsum;
               }
           }
           ccf[i] = tsum;
       }
       max = WhereMax(ccf, ntemp);
       double ccf_max_left = max[1];
       double ccf_shift_left = max[0]*precision;
       free(xtemp);
       free(ytemp);
       free(yupsampled);
       free(ccf);
       free(max);
       if(ccf_max_left>ccf_max_right){
          return ccf_shift_left-mutemp;
       }
       return mutemp+ccf_shift_right;
}
