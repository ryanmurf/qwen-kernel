import unittest

from audit_native_decode_loads import ORDER, SIZES, compare


def matrix():
    runs=[]
    for index,mode in enumerate(ORDER):
        rows=[dict(type='metadata',mode=mode,launch=index,context=32768,sizes=list(SIZES),
                   fixture_sha256='a'*64,prompt_sha256={str(n):'b'*64 for n in SIZES},
                   harness_sha256='c'*64,protocol='fresh')]
        for n in SIZES:
            count=512 if n>=16384 else 128
            speed=20 if mode=='loads' else 10
            rows.append(dict(type='decode',prompt_tokens=n,exact_output_length=True,
                             coherent_counting_prefix=True,streamed_tokens=count,requested_output_tokens=count,
                             output_token_sha256='d'*64,output_text_sha256='e'*64,
                             ttft_seconds=10,decode_tokens_per_second=speed,stream_total_seconds=10+count/speed))
        rows.append(dict(type='result',result='PASS',cases=6))
        runs.append(rows)
    return runs


class DecodeAuditTests(unittest.TestCase):
    def test_counterbalanced_winner(self):
        result=compare(matrix())
        self.assertTrue(result['loads_promoted'])
        self.assertEqual(len(result['summary']),12)

    def test_missing_reordered_or_changed_output_rejected(self):
        for mutate in (lambda x:x.pop(),lambda x:x[1].pop(2),
                       lambda x:x[1][1].update(output_token_sha256='f'*64),
                       lambda x:x[1][0].update(mode='serial'),
                       lambda x:x[1][1].update(decode_tokens_per_second=float('nan')),
                       lambda x:x[1][1].update(streamed_tokens=127),
                       lambda x:x[1][0].update(protocol='cached')):
            data=matrix()
            mutate(data)
            with self.assertRaises(ValueError): compare(data)

    def test_small_decode_regression_prevents_promotion(self):
        data=matrix()
        for i in (1,2): data[i][1]['decode_tokens_per_second']=9.6
        self.assertFalse(compare(data)['loads_promoted'])

    def test_prefill_regression_prevents_promotion(self):
        data=matrix()
        for i in (1,2): data[i][4]['ttft_seconds']=10.4
        self.assertFalse(compare(data)['loads_promoted'])

    def test_both_long_contexts_must_gain(self):
        data=matrix()
        for i in (1,2): data[i][6]['decode_tokens_per_second']=12.4
        self.assertFalse(compare(data)['loads_promoted'])


if __name__=='__main__': unittest.main()
