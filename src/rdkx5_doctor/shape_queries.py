"""Saved JSON queries; never reload the source ONNX."""
from .text_utils import json_text

def register_shape_commands(sub):
    for command in ('shapes','shape'):
        c=sub.add_parser(command,help='有界 Shape 证明与未知来源（读取保存的 JSON）')
        c.add_argument('--analysis',required=True,type=__import__('pathlib').Path)
        c.add_argument('--json',action='store_true')
        if command=='shape':c.add_argument('--tensor',required=True)
        else:
            c.add_argument('--summary',action='store_true')
            c.add_argument('--status',choices=['PROVEN','PARTIAL','UNKNOWN','CONFLICT'])
            c.add_argument('--limit',type=int,default=20)

def run_shape_query(args,data,display):
    section=data.get('shape_analysis')
    if not section:raise ValueError('旧报告没有 Shape 溯源；请重新 analyze 获取 schema 1.3')
    if args.command=='shapes':
        if args.limit<0:raise ValueError('limit 必须非负')
        facts=[f for f in section['facts'] if not args.status or f['status']==args.status]
        result=section['summary'] if args.summary else {'total':len(facts),'facts':facts if args.limit==0 else facts[:args.limit],'omitted':max(0,len(facts)-args.limit) if args.limit else 0}
    else:
        matches=[f for f in section['facts'] if f['tensor_name']==args.tensor]
        if not matches:raise ValueError('未找到 Tensor；请使用 shapes 查询')
        result={'fact':matches[0],'proofs':[p for p in section['proofs'] if p['tensor_name']==args.tensor],
                'unknown_origins':[o for o in section['unknown_origins'] if o['tensor_name']==args.tensor]}
    print(json_text(result,ensure_ascii=False,indent=2))
    return 0
