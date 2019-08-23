#include <stdio.h>
#include <locale.h>
#include <time.h>

#include <sieknet.h>
#include <tensor.h>

size_t clock_us(){
  struct timespec start;

  clock_gettime(CLOCK_REALTIME, &start);
  return start.tv_sec * 1e6 + start.tv_nsec / 1e3;
}

int main(int argc, char **argv){
	if(argc < 2)
		SK_ERROR("Please provide a .sk file.");

  srand(1271998);
  setlocale(LC_NUMERIC, "");
  setbuf(stdout, NULL);

	/*
	 * Note: for small values of t, underflow may cause unusually large relative errors.
	 */
	const size_t t = 65;
	const double epsilon = 2e-2;
	const double threshold = 1e-4;

	Network n = sk_create_network(argv[1]);
	Tensor x = create_tensor(SIEKNET_CPU, t, n.input_dimension);
	Tensor y = create_tensor(SIEKNET_CPU, t, n.layers[n.depth-1]->output.dims[1]);
	tensor_fill_random(x, 0, 1);
	tensor_fill_random(y, 0.5, 0.12);

  SK_COST_FN cost_fn = SK_QUADRATIC_COST;

	printf("Checking %'lu parameters...\n", n.num_params);
	double norm = 0;
	size_t count = 0;
	float *params = tensor_raw(n.params);
	float *p_grad = tensor_raw(n.param_grad);
	sk_forward(&n, x);
	sk_cost(n.layers[n.depth-1], y, cost_fn);
	sk_backward(&n);
	sk_wipe(&n);

  size_t start = clock_us();
	for(size_t i = 0; i < n.num_params; i++){
    printf("Checked %'9lu of %'9lu parameters in %3.2fs\t\r", i, n.num_params, 1e-6*(clock_us() - start));
		double predicted_grad = p_grad[i];

		params[i] += epsilon;

		sk_forward(&n, x);
		double c1 = sk_cost(n.layers[n.depth-1], y, cost_fn);
		n.t = 0;
		sk_wipe(&n);

		params[i] -= 2 * epsilon;
		sk_forward(&n, x);
		double c2 = sk_cost(n.layers[n.depth-1], y, cost_fn);
		n.t = 0;
		sk_wipe(&n);

		double empirical_grad = (c1 - c2) / (2 * epsilon);
		double diff = (predicted_grad - empirical_grad);
		double relative = (fabs(diff) / MAX(fabs(predicted_grad), fabs(empirical_grad)));
    printf("%f vs %f\n", predicted_grad, empirical_grad);
    //
    if(isnan(relative)){
      printf("\n(warning: zero parameter)\n");
      continue;
    }

		if(relative/t > 1e-3){
			Layer *culprit;
			for(int j = 0; j < n.depth; j++){
				Layer *l = n.layers[j];
				if(l->param_idx <= i && l->param_idx + l->num_params > i){
					culprit = l;
					break;
				}
			}
			printf("param %'9lu: abs. err: %9.8f, relative err: %4.3f, est. grad %10.8f, measured grad %10.8f, culprit '%s'\n", i, fabs(diff)/t, relative/t, predicted_grad, empirical_grad, culprit->name);
		}
		norm += relative;
		count++;
		params[i] += epsilon;

	}
  printf("\n");
	tensor_fill(n.param_grad, 0.0f);
	norm /= count * t;
	if(norm < threshold)
		printf("PASSED (norm %12.11f)\n", norm);
	else
		printf("FAILED (norm %12.11f)\n", norm);

  return 0;
}