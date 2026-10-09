from pathlib import Path
import subprocess, json, time, sys, os

root = Path('/home/xhm/lianghua_ws/skillzuoye')
out = Path('/tmp/rdkx5-validation-dir').read_text().strip()
out = Path(out)
python = str(root / '.venv/bin/python')
history = out / 'command_log.jsonl'

def run(label, args):
    started = time.monotonic()
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    p = subprocess.run(args, cwd=root, text=True, capture_output=True, env=env)
    (out / (label + '.stdout')).write_text(p.stdout)
    (out / (label + '.stderr')).write_text(p.stderr)
    record = dict(label=label, argv=args, unset_env=['PYTHONPATH'], returncode=p.returncode,
                  elapsed_seconds=time.monotonic()-started)
    with history.open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False)+'\n')
    print(label, 'exit', p.returncode, 'seconds', round(record['elapsed_seconds'],3))
    print((p.stdout+p.stderr)[:5000])
    return p

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'baseline':
        for label, args in [
            ('help', [python,'-m','rdkx5_doctor','--help']),
            ('rules_validate',[python,'-m','rdkx5_doctor','rules','validate']),
            ('pytest_isolated',[str(root/'.venv/bin/pytest'),'-q']),
            ('environment',[python,'-m','pip','freeze']),
            ('pip_check',[python,'-m','pip','check'])]:
            p=run(label,args)
            if p.returncode and label != 'pip_check': sys.exit(p.returncode)
    elif mode == 'analyze':
        model=json.loads((out/'validation_metadata.json').read_text())['model_path']
        p=run('analyze',[python,'-m','rdkx5_doctor','analyze','--model',model,'--out',str(out)])
        sys.exit(p.returncode)
