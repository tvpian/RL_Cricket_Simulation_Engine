"""Export an offline, selectable delivery replay; no running server required."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from cricket_rl import CricketEnv
from cricket_rl.experiment import baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--action', type=int, choices=range(9), default=4)
    parser.add_argument('--deliveries', type=int, default=6)
    parser.add_argument('--policy', choices=('manual', 'heuristic', 'random'), default='manual')
    parser.add_argument('--pitch', choices=('standard', 'green', 'dry'), default='standard')
    parser.add_argument('--style', choices=('defensive', 'balanced', 'aggressive', 'unseen'), default='balanced')
    parser.add_argument('--static-batting', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('runs/replay.html'))
    args = parser.parse_args()
    if not 1 <= args.deliveries <= 300 or args.seed < 0:
        parser.error('Need 1–300 delivery attempts and a nonnegative seed')
    env = CricketEnv(record=True, styles=(args.style,), pitch=args.pitch,
                     balls=min(300, max(6, args.deliveries)), max_wickets=10, adaptive_batting=not args.static_batting)
    env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    deliveries = []
    for _ in range(args.deliveries):
        action = args.action if args.policy == 'manual' else baseline(env, args.policy, rng)
        _, reward, terminated, truncated, info = env.step(action)
        deliveries.append(dict(trace=env.last_delivery.trajectory,
                               fielders=env.last_delivery.fielder_trajectory,
                               info=info, reward=reward, action=int(action)))
        if terminated or truncated:
            break
    data = json.dumps(dict(deliveries=deliveries, pitch=asdict(env.pitch), seed=args.seed))
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cricket Strategy Lab — replay</title>
<style>body{background:#0a141c;color:#e9f3f1;font:16px system-ui;max-width:1000px;margin:30px auto;padding:20px}h1{color:#b6f786}canvas{background:#12382c;border-radius:14px;width:100%}button,input,select{margin:6px;padding:9px;border:1px solid #355261;border-radius:6px;background:#10232e;color:inherit}p{color:#b6cad2;line-height:1.6}#timeline{width:95%}.active{border-color:#b6f786}</style>
<h1>Cricket Strategy Lab</h1><p>Offline delivery replay · ball flight, bat contact, and fielder pursuit</p>
<div id="deliveries"></div><h2 id="score"></h2><canvas id="field" width="1000" height="650"></canvas>
<div><button id="play">Pause / play</button><button id="restart">Restart</button><select id="view" aria-label="Camera"><option value="broadcast">Broadcast</option><option value="top">Top down</option><option value="side">Bounce / side view</option></select><label>Speed <input id="speed" type="range" min="0.25" max="4" value="1" step="0.25"></label></div>
<input id="timeline" type="range" min="0" max="1" value="0" step="1" aria-label="Timeline"><p id="telemetry"></p><p>Blue: moving fielders. Yellow: ball. Shadow: ground projection. Coordinates are in metres. Illustrative, uncalibrated cricket physics. Exported replay is self-contained and requires no network connection.</p>
<script>
const data=DATA;
const $=id=>document.getElementById(id),ctx=$('field').getContext('2d');
let selected=0, frame=0, playing=true, last=0;
function project(p){if($('view').value==='side')return [260+p[1]*9,500-p[2]*60];if($('view').value==='top')return [500+p[0]*3.5,325-p[1]*3.5];return [500+p[0]*4,350-p[1]*2.8-p[2]*14];}
function choose(index){selected=index;frame=0;playing=true;const d=data.deliveries[index],i=d.info;$('score').textContent=`${i.runs}/${i.wickets} · ${i.overs} overs · ${i.extra_type||i.event}`;$('timeline').max=d.trace.length-1;[...$('deliveries').children].forEach((b,k)=>b.className=k===index?'active':'');}
data.deliveries.forEach((d,i)=>{const b=document.createElement('button');b.textContent=`${i+1}: ${d.info.wicket?'W':d.info.delivery_runs}`;b.onclick=()=>choose(i);$('deliveries').append(b);});
function circle(p,r,c){let [x,y]=project(p);ctx.fillStyle=c;ctx.beginPath();ctx.arc(x,y,r,0,7);ctx.fill();}
function draw(now){const elapsed=last?Math.min((now-last)/1000,.1):0;last=now;const d=data.deliveries[selected];if(playing)frame=Math.min(d.trace.length-1,frame+elapsed*50*Number($('speed').value));const idx=Math.floor(frame);$('timeline').value=idx;ctx.clearRect(0,0,1000,650);
ctx.strokeStyle='#b8d7bc80';for(const r of [30,data.pitch.boundary]){ctx.beginPath();for(let i=0;i<=100;i++){let a=i/100*Math.PI*2,p=project([Math.sin(a)*r,Math.cos(a)*r,0]);i?ctx.lineTo(...p):ctx.moveTo(...p);}ctx.stroke();}
ctx.fillStyle='#bfa87a';ctx.beginPath();[[-1.5,-1.22,0],[1.5,-1.22,0],[1.5,20.12,0],[-1.5,20.12,0]].forEach((p,i)=>i?ctx.lineTo(...project(p)):ctx.moveTo(...project(p)));ctx.closePath();ctx.fill();
for(const f of d.fielders[idx])circle([...f,0],5,'#7bc2ff');circle([.5,0,0],6,'#ffebbd');
ctx.strokeStyle='#ffffff90';ctx.beginPath();for(let i=0;i<=idx;i++){let p=project(d.trace[i]);i?ctx.lineTo(...p):ctx.moveTo(...p);}ctx.stroke();const p=d.trace[idx];circle([p[0],p[1],0],5,'#00000070');circle(p,5,'#fff26b');$('telemetry').textContent=`Seed ${data.seed} · ${(idx*.02).toFixed(2)} seconds · ball height ${p[2].toFixed(2)} m · action ${d.action} · batter readiness ${Math.round(d.info.readiness*100)}%`;requestAnimationFrame(draw);}
$('play').onclick=()=>playing=!playing;$('restart').onclick=()=>{frame=0;playing=true;};$('timeline').oninput=()=>{frame=Number($('timeline').value);playing=false;};choose(0);requestAnimationFrame(draw);
</script></html>'''.replace('DATA', data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    env.close()
    print(args.output)


if __name__ == '__main__':
    main()
