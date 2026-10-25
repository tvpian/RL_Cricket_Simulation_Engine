/* NumPy and the repository wheel execute inside Pyodide, off the UI thread. */
const ready = (async()=>{
  postMessage({progress:'Loading Python · first visit may take a minute…'});
  importScripts('./runtime/pyodide.js');
  const py=await loadPyodide({indexURL:new URL('./runtime/',self.location.href).href});
  await py.loadPackage(['numpy','micropip']);
  postMessage({progress:'Loading cricket environment…'});
  const base=new URL('./',self.location.href).href;
  const wheels=await (await fetch(base+'dependency-wheels.json')).json();
  py.globals.set('dependency_wheels',JSON.stringify(wheels.map(name=>base+name)));
  py.globals.set('wheel_url',base+'cricket_strategy_rl-0.1.0-py3-none-any.whl');
  await py.runPythonAsync(`
import micropip
import json
await micropip.install(json.loads(dependency_wheels), deps=False)
await micropip.install(wheel_url, deps=False)
import json
from cricket_rl.session import Session
session = Session()
def browser_request(raw):
    request = json.loads(raw)
    path, body = request['path'], request.get('body') or {}
    if path == 'state': result = session.state()
    elif path == 'reset': result = session.reset(body)
    elif path == 'bowl': result = session.bowl(body)
    elif path == 'field':
        session.env.set_field(body.get('fielders'))
        result = session.state()
    else: raise ValueError('Unsupported browser operation')
    return json.dumps(result)
`);
  return py;
})();
let queue=Promise.resolve();
self.onmessage=({data})=>{
  queue=queue.then(async()=>{
    try{
      const py=await ready;
      py.globals.set('request_json',JSON.stringify(data));
      const result=JSON.parse(py.runPython('browser_request(request_json)'));
      postMessage({id:data.id,result});
    }catch(error){postMessage({id:data.id,error:'Browser environment: '+error.message});}
  });
};
