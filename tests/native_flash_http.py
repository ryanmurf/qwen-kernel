#!/usr/bin/env python3
"""Real Flash HTTP checks; requires an exclusive serving test window."""
import argparse
import concurrent.futures
import http.client
import json
import re
import time
import urllib.error
import urllib.request
import urllib.parse

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url",default="http://127.0.0.1:8194")
    p.add_argument("--benchmark-only",action="store_true")
    p.add_argument("--cancel-only",action="store_true")
    p.add_argument("--skip-cancel",action="store_true",help="skip direct-engine disconnect check when testing a buffering proxy")
    p.add_argument("--prefill-tokens",type=int,default=0,help="optional one-output-token prompt-processing timing")
    p.add_argument("--prefill-diverse",action="store_true",help="use distinct token ids for the prefill benchmark (defeats the PLE row cache like real text)")
    args=p.parse_args()
    def open_request(path,body):
        req=urllib.request.Request(args.url+path,data=json.dumps(body).encode(),
                                   headers={"Content-Type":"application/json"})
        return urllib.request.urlopen(req,timeout=240)
    def post(path,body):
        with open_request(path,body) as response: return json.load(response)
    def record(name,**fields): print(json.dumps({"test":name,**fields}),flush=True)
    def fixed(tokens,expected):
        out=post("/completion",{"prompt":tokens,"n_predict":1,"temperature":0,"return_tokens":True})
        assert out["tokens"]==[expected],out
        return out
    def cancellation():
        # Close the TCP request without waiting for response headers or the
        # first SSE event; the disconnect must interrupt the prefill itself.
        address=urllib.parse.urlsplit(args.url)
        connection=http.client.HTTPConnection(address.hostname,address.port,timeout=10)
        connection.request("POST","/completion",body=json.dumps({"prompt":[9419]*768,
                           "n_predict":1024,"stream":True,"temperature":0}),
                           headers={"Content-Type":"application/json"})
        time.sleep(0.2)
        start=time.monotonic()
        connection.close()
        fixed([198],36)
        elapsed=time.monotonic()-start
        record("prefill_cancel_then_request",result="PASS" if elapsed<5 else "SLOW",seconds=elapsed)
        assert elapsed<5, "disconnected prefill did not release the slot promptly"
    if args.cancel_only:
        fixed([198],36)
        cancellation()
        return
    if not args.benchmark_only:
        fixed(list(range(198,214)),271)
        record("teacher_forced_http",result="PASS")
    prompt=("<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\n"
            "Write the integers from 1 to 40, separated by commas.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    prompt_tokens=len(post("/tokenize",{"content":prompt,"add_special":True})["tokens"])
    start=time.monotonic(); stamps=[]; chunks=[]; token_counts=[]; final={}
    with open_request("/completion",{"prompt":prompt,"n_predict":96,"temperature":0,
                                     "stream":True,"return_tokens":True,"cache_prompt":False}) as response:
        for line in response:
            if not line.startswith(b"data: "): continue
            raw=line[6:].strip()
            if raw==b"[DONE]": break
            item=json.loads(raw)
            if item.get("stop"):
                final=item; continue
            if item.get("content") or item.get("tokens"):
                stamps.append(time.monotonic()); chunks.append(item.get("content",""))
                token_counts.append(len(item.get("tokens",[])))
    assert len(stamps)>2,final
    counting=re.sub(r"\s+","","".join(chunks))
    coherent=bool(counting) and ",".join(str(i) for i in range(1,41)).startswith(counting)
    record("stream_benchmark",prompt_tokens=prompt_tokens,events=len(stamps),
           first_token_seconds=stamps[0]-start,total_seconds=time.monotonic()-start,
           inter_event_per_second=(len(stamps)-1)/(stamps[-1]-stamps[0]),
           streamed_tokens=sum(token_counts),
           decode_tokens_per_second=(sum(token_counts)-token_counts[0])/(stamps[-1]-stamps[0]) if all(token_counts) else None,
           coherent_counting_prefix=coherent,
           content="".join(chunks),reported_timings=final.get("timings"))
    if args.prefill_tokens:
        assert 1<=args.prefill_tokens<=4096
        prompt_ids=[1000+i for i in range(args.prefill_tokens)] if args.prefill_diverse else [9419]*args.prefill_tokens
        start=time.monotonic()
        out=post("/completion",{"prompt":prompt_ids,"n_predict":1,
                 "temperature":0,"return_tokens":True,"cache_prompt":False})
        elapsed=time.monotonic()-start
        record("prefill_benchmark",prompt_tokens=args.prefill_tokens,diverse=args.prefill_diverse,seconds=elapsed,
               prompt_tokens_per_second=args.prefill_tokens/elapsed,
               reported_timings=out.get("timings"),output_tokens=out.get("tokens"))
    if args.benchmark_only: return
    assert coherent, "counting completion was not a coherent prefix"
    simple={"model":"native-flash","max_tokens":512,"temperature":0,
            "messages":[{"role":"user","content":"Reply with exactly READY and nothing else."}]}
    result=post("/v1/messages",simple)
    text="".join(b.get("text","") for b in result["content"])
    assert text.strip()=="READY" and result["stop_reason"]=="end_turn",result
    record("claude_text",result="PASS",usage=result["usage"])
    tools=[{"name":"echo","description":"Echo the supplied text.","input_schema":{
        "type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}]
    tool_req={"model":"native-flash","max_tokens":768,"temperature":0,"tools":tools,
              "system":"Use the echo tool exactly once when asked. Do not answer directly.",
              "messages":[{"role":"user","content":"Call echo with text READY."}]}
    result=post("/v1/messages",tool_req)
    calls=[b for b in result["content"] if b["type"]=="tool_use"]
    assert result["stop_reason"]=="tool_use" and len(calls)==1,result
    assert calls[0]["name"]=="echo" and calls[0]["input"]=={"text":"READY"},result
    record("claude_xml_tool",result="PASS",call=calls[0],usage=result["usage"])
    tool_req["system"]="After a tool result, reply with its text only."
    tool_req["messages"] += [{"role":"assistant","content":result["content"]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":calls[0]["id"],"content":"READY"}]}]
    result=post("/v1/messages",tool_req)
    text="".join(b.get("text","") for b in result["content"])
    assert text.strip()=="READY" and result["stop_reason"]=="end_turn",result
    record("claude_tool_roundtrip",result="PASS",usage=result["usage"])
    events=[]
    with open_request("/v1/messages",{**simple,"stream":True}) as response:
        for line in response:
            if line.startswith(b"data: "): events.append(json.loads(line[6:]))
    assert events[0]["type"]=="message_start" and events[-1]["type"]=="message_stop",events
    text="".join(e.get("delta",{}).get("text","") for e in events)
    assert text.strip()=="READY",events
    record("claude_stream",result="PASS",events=len(events))
    if not args.skip_cancel:
        cancellation()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(fixed,[198],36),pool.submit(fixed,list(range(198,214)),271)]
        for job in jobs: job.result()
    record("concurrent_isolation",result="PASS")
    try:
        post("/completion",{"prompt":[248320],"n_predict":1})
        raise AssertionError("invalid token accepted")
    except urllib.error.HTTPError as err: assert err.code==400
    record("invalid_token",result="PASS")

if __name__=="__main__": main()
