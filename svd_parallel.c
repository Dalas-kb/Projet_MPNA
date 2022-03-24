#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

double moy(double *a, int n){
        double m = 0.0;
        for(int i = 0; i<n; i++)
                m += a[i];
        return m / (double)n;
}

double** lect_mat(const int m, const int n){
	double** a = malloc(sizeof(double*) * m);


		#pragma omp parallel for
		for(int i=0; i<m; i++)
			a[i] = malloc(sizeof(double) * n);

		for(int i=0; i<m; i++){
			#pragma omp parallel for
			for(int j=0; j<n; j++){
				a[i][j] = (rand() % RAND_MAX)/10e5;
			}
		}	
	
	return a;
}


void affich_mat(const int m, const int n, double** A){
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++)
			printf("%lf\t", A[i][j]);
		printf("\n");
	}
}

double* prod_mat_vect(double** A, double** v, const int m, const int n, const int k){
	double* col_tmp = calloc(m, sizeof(double));
	for(int i=0; i<m; i++)
	#pragma omp parallel for
		for(int j=0; j<n; j++){
			col_tmp[i] += A[i][j]*v[j][k];
		}
	return col_tmp;
}


double* prod_mat_vect_t(double** A, double** u, const int m, const int n, const int k){
	double* col_tmp = calloc(n, sizeof(double));
	for(int i=0; i<n; i++)
	#pragma omp parallel for
		for(int j=0; j<m; j++)
			col_tmp[i] += A[j][i]*u[i][k];
	return col_tmp;
}


double norme(double** u, const int k, const int m){
	double norm = 0.;
	#pragma omp parallel for
	for(int i=0; i<m; i++){
		double tmp = u[i][k];
		norm += tmp*tmp;
	}
	norm = sqrt(norm);
	return norm;
}


void calc_u(const int k, double** A, const int m, const int n ,double** v, double*** u, double* beta){
	double* col_tmp = malloc(sizeof(double) * m);
	col_tmp = prod_mat_vect(A, v, m, n, k);	
	for(int j=0; j<m; j++)
	for(int i=0; i<m; i++){
		if(k == 0)
			(*u)[i][k] = col_tmp[i];
		else
			(*u)[i][k] = col_tmp[i] - beta[k-1] * (*u)[i][k-1];
	}
}


void calc_v(const int k, double** A, const int m, const int n, double*** v, double** u, double* alpha){
	double* col_tmp = malloc(sizeof(double) * n);
	col_tmp = prod_mat_vect_t(A, u, m, n, k);
	for(int i=0; i<n; i++)
		(*v)[i][k+1] = col_tmp[i] - alpha[k] * (*v)[i][k];
}


void algo_golub(double** A, const int m, const int n, double** alpha, double** beta, double*** u, double*** v){
	for(int i=0; i<n; i++){
		if (i == 0)
			(*v)[i][0] = 1;
		else
			(*v)[i][0] = 0;
	}
	

	for(int k = 0; k < n; k++){
		calc_u(k, A, m, n, *v, u, *beta);
		(*alpha)[k] = norme(*u, k, m);
		#pragma omp parallel for
		for(int i=0; i<m; i++){
			double tmp_u = (*u)[i][k];
			assert((*alpha)[k] != 0);
			(*u)[i][k] = tmp_u / (*alpha)[k];
		}
		
		calc_v(k, A, m, n, v, *u, *alpha);
		(*beta)[k] = norme(*v, k+1, n);
		#pragma omp parallel for
		for(int i=0; i<n; i++){
			double tmp_u = (*v)[i][k+1];
			assert((*beta)[k] != 0);
			(*v)[i][k+1] = tmp_u / (*beta)[k];
		}
	}
}


void prod_norm(const int n, double** alpha, double** beta, double** diag_t, double** sdiag_t){
	double alp = (*alpha)[0];
	double bet = 0., alp_ = 0.;
	(*diag_t)[0] = 	alp * alp;	
	for(int i=1; i<n; i++){
		alp = (*alpha)[i];
		alp_= (*alpha)[i-1];

		bet = (*beta)[i-1] ;

		(*diag_t)[i] = (alp * alp) + (bet * bet);
		(*sdiag_t)[i-1]= alp_ * bet;
	}
}


void prod_mat_mat(double **U, double** Ua, const int m, const int n, double** Ub, const int mm, const int nn){

	for(int i=0; i<m; i++){
       		for(int j=0; j<n; j++){
			for(int k=0; k<n; k++)
				U[i][j] = Ua[i][k]*Ub[k][i];
		}
	}	
}


void prod_mat_mat_1d(double** V, double* z, const int n, const int nn, double** v, const int m, const int mm){

 	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
				V[i][j] = z[i + j * n] * v[i][j];
		}
	}
}

int main(int argc, char** argv){
	if(argc != 5){
		perror("nbre arguments incorrect!\n");
		exit(0);
	}
	
	clock_t start, end;
	
	double avg_time;
	
	int m = atoi(argv[1]), n = atoi(argv[2]), n_reps = atoi(argv[3]);
	int b_print = atoi(argv[4]);
	double samples[n_reps];
	double** A;
	A = lect_mat(m, n);
	
	
	double** v = malloc(sizeof(double*) * (n+1));
	double** u = malloc(sizeof(double*) * m);
	
	#pragma omp for
	for(int i=0; i<n+1; i++)
		v[i] = malloc(sizeof(double*) * (n+1));
	
	#pragma omp for
	for(int i=0; i<m; i++)
		u[i] = malloc(sizeof(double*) * n);

	double* alpha = malloc(sizeof(double) * n);      
	double* beta  = malloc(sizeof(double) * (n - 1));
	
	double* diag_t = malloc(sizeof(double) * n);
	double* sdiag_t= malloc(sizeof(double) * (n-1));
	
	double vl = 0., vu = 0.;
	lapack_int il, iu, mm, ldz, info;
	double abstol = 10e-8;
	
	ldz = n;
	double* w =  malloc(sizeof(double) * n );
	double* z =  malloc(sizeof(double) * n * n);
	
	lapack_int* ifail = (lapack_int*)malloc(sizeof(lapack_int) * n * 2);

	double* d = malloc(sizeof(double) * n);
	double* e = malloc(sizeof(double) * (n-1));
	
	double** Ub = malloc(sizeof(double*) * n);
	for(int i=0; i<n; i++)
		Ub[i] = malloc(sizeof(double) * n);

	double* Sigma = malloc(sizeof(double) * n);
	double ww;
	
	double** B_Vb = malloc(sizeof(double*) * n);
	for(int i=0; i<n; i++)
		B_Vb[i] = malloc(sizeof(double) * n);
		
	double** U = malloc(sizeof(double) * m * n);
	for(int i=0; i<m; i++)
		U[i] = malloc(sizeof(double) * n);
	double** V = malloc(sizeof(double) * n * n);
	for(int j=0; j<n; j++)
		V[j] = malloc(sizeof(double) * n);
		
	for(int k = 0; k<n_reps; k++){
	
		start = clock();
	
		algo_golub(A, m , n,  &alpha, &beta, &u, &v);
		
		prod_norm(n, &alpha, &beta, &diag_t, &sdiag_t);
		
		d = diag_t;
		e = sdiag_t; 

		info = LAPACKE_dstevx(LAPACK_ROW_MAJOR, 'V', 'A', n, d, e, vl, vu, il, iu, abstol, &mm, w, z, ldz, ifail);

		for(int i=0; i<n; i++){
			ww = w[i];
			Sigma[i] = sqrt(ww);
		}
		
		for(int i=0; i<n; i++){
			for(int j=0; j<n; j++){
				if (i == n-1){
					B_Vb[n - 1][j] = alpha[n-1] * z[j + (n - 1) * n];
					break;
				}
				B_Vb[i][j] = alpha[i] * z[j + i * n] + beta[i] * z[(j + i * n) + 1];
			}
		}

		for(int i=0; i<n; i++)
			for(int j=0; j<n; j++){
				Ub[i][j] = B_Vb[i][j] / (Sigma[j]);
			}


		

		prod_mat_mat(U, u, m, n, Ub, n, n);
		


		prod_mat_mat_1d(V, z, n, n, v, n, n);
		

		
		end = clock();
		samples[k] = (float)(end - start) / CLOCKS_PER_SEC;
	}
	
	if (b_print>0){
	
		printf("Matrice A:\n");
		affich_mat(m, n, A);
		printf("\n");
		
		printf("1) Bidiagonalisation de la matrice: \n");
		
		printf("elements diagonaux: \n");
		for(int i=0; i<n; i++){
			printf("%lf\t", alpha[i]);
		}
		printf("\n");
		printf("elements surdiagonaux: \n");
		for(int i=0; i<n-1; i++)
			printf("%lf\t", beta[i]);
		printf("\n");
		
		printf("u: \n");
		for(int i=0; i<m; i++){
			for(int j=0; j<n; j++)
				printf("%lf\t",u[i][j]);
			printf("\n");
		}
		printf("v: \n");
		for(int i=0; i<n + 1; i++){
			for(int j=0; j<n; j++)
				printf("%lf\t",v[i][j]);
			printf("\n");
		}
		printf("\n");
		
		printf("2) Decomposition spectrale: \n");
		printf("matrice tridiagonale : \n");
		for(int i=0; i<n; i++)
			printf("%lf\t", diag_t[i]);
		printf("\n");
		printf("elements symetriques: \n");
		for(int i=0; i<n-1; i++)
			printf("%lf\t", sdiag_t[i]);
		printf("\n");
		
		
		for(int i=0; i<n; i++){
			printf("val sing %d = %lf\n", i+1, Sigma[i]);
		}
		
		printf("U: \n");
		affich_mat(m, n, U);
		
		printf("V: \n");
		affich_mat(n, n, V);
	}
	printf("\n");
	avg_time  = moy(samples, n_reps)*1000;
	printf("temps moy : %.2fms\n", avg_time);
}
