#include <stddef.h>

#include <nnpack.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>

#include <nnpack/hwinfo.h>
#include <nnpack/softmax.h>
#include <nnpack/validation.h>


struct NNP_CACHE_ALIGN softmax_context {
	nnp_softmax_function softmax_function;
	size_t channels;
	const float* input;
	float* output;
};

static void compute_softmax_output(
	const struct softmax_context context[RESTRICT 1],
	size_t sample)
{
	const nnp_softmax_function softmax = context->softmax_function;
	const size_t channels              = context->channels;

	//const float (*input)[channels] = (const float(*)[channels]) context->input;
	//float (*output)[channels] = (float(*)[channels]) context->output;

	softmax(channels, (const float *)context->input + sample*channels,
                     (float *)context->output + sample+channels);
}

struct NNP_CACHE_ALIGN inplace_softmax_context {
	nnp_inplace_softmax_function softmax_function;
	size_t channels;
	float* data;
};

static void compute_inplace_softmax_output(
	const struct inplace_softmax_context context[RESTRICT 1],
	size_t sample)
{
	const nnp_inplace_softmax_function softmax = context->softmax_function;
	const size_t channels                      = context->channels;

	//float (*data)[channels] = (float(*)[channels]) context->data;

	softmax(channels, (float *)context->data + channels*sample);
}

enum nnp_status nnp_softmax_output(
	size_t batch_size,
	size_t channels,
	const float* input,
	float* output,
	pthreadpool_t threadpool)
{
	enum nnp_status status = validate_softmax_arguments(batch_size, channels);
	if (status != nnp_status_success) {
		return status;
	}

	if (input != output) {
		/* Out-of-place softmax */
		struct softmax_context softmax_context = {
			.softmax_function = nnp_hwinfo.activations.softmax,
			.channels = channels,
			.input = input,
			.output = output,
		};
		pthreadpool_compute_1d(threadpool,
			(pthreadpool_function_1d_t) compute_softmax_output,
			&softmax_context,
			batch_size);
	} else {
		/* In-place softmax */
		struct inplace_softmax_context inplace_softmax_context = {
			.softmax_function = nnp_hwinfo.activations.inplace_softmax,
			.channels = channels,
			.data = output,
		};
		pthreadpool_compute_1d(threadpool,
			(pthreadpool_function_1d_t) compute_inplace_softmax_output,
			&inplace_softmax_context,
			batch_size);
	}

	return nnp_status_success;
}
