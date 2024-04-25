# import random
# from http import HTTPStatus
# import dashscope

# # sk-5a83da0541c643e0823d0d2056837bed

# def call_with_messages():
#     content = '''Assuming you are an experienced code vulnerability analyst and the following code may have vulnerabilities.Is the code vulnerable?(YES/NO) static inline void do_imdct(AC3DecodeContext *s, int channels)\n\n{\n\n    int ch;\n\n\n\n    for (ch=1; ch<=channels; ch++) {\n\n        if (s->block_switch[ch]) {\n\n            int i;\n\n            float *x = s->tmp_output+128;\n\n            for(i=0; i<128; i++)\n\n                x[i] = s->transform_coeffs[ch][2*i];\n\n            ff_imdct_half(&s->imdct_256, s->tmp_output, x);\n\n            s->dsp.vector_fmul_window(s->output[ch-1], s->delay[ch-1], s->tmp_output, s->window, s->add_bias, 128);\n\n            for(i=0; i<128; i++)\n\n                x[i] = s->transform_coeffs[ch][2*i+1];\n\n            ff_imdct_half(&s->imdct_256, s->delay[ch-1], x);\n\n        } else {\n\n            ff_imdct_half(&s->imdct_512, s->tmp_output, s->transform_coeffs[ch]);\n\n            s->dsp.vector_fmul_window(s->output[ch-1], s->delay[ch-1], s->tmp_output, s->window, s->add_bias, 128);\n\n            memcpy(s->delay[ch-1], s->tmp_output+128, 128*sizeof(float));\n\n        }\n\n    }\n\n}\n\nYour answer should either be 'YES' or 'NO' only.'''
#     messages = [
#         {'role': 'user', 'content': content}]
#     response = dashscope.Generation.call(
#         'qwen1.5-72b-chat',
#         messages=messages,
#         # set the random seed, optional, default to 1234 if not set
#         seed=random.randint(1, 10000),
#         result_format='message',  # set the result to be "message" format.
#     )
#     if response.status_code == HTTPStatus.OK:
#         print(response)
#     else:
#         print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
#             response.request_id, response.status_code,
#             response.code, response.message
#         ))


# if __name__ == '__main__':
#     call_with_messages()


import random
from http import HTTPStatus
import dashscope


def call_with_messages():
    messages = [
        {'role': 'user', 'content': 'Please help me with the code vulnerability analysis.'},]
    response = dashscope.Generation.call(
        'codeqwen1.5-7b-chat',
        messages=messages,
        # set the random seed, optional, default to 1234 if not set
        seed=random.randint(1, 10000),
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


if __name__ == '__main__':
    call_with_messages()