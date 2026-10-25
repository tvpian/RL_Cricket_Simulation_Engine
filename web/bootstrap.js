/* Worker transport for the same Python simulator used by the server. */
const worker = new Worker('./python-worker.js');
const pending = new Map();
let sequence = 0;
let failure = null;
worker.onmessage = ({data}) => {
  if (data.progress) {
    const status=document.getElementById('connection');
    if(status)status.textContent=data.progress;
    return;
  }
  const task=pending.get(data.id);
  if(!task)return;
  pending.delete(data.id);
  data.error ? task.reject(new Error(data.error)) : task.resolve(data.result);
};
worker.onerror = event => {
  failure=new Error('The browser Python runtime could not start. Check your connection and try again.');
  for(const task of pending.values())task.reject(failure);
  pending.clear();
};
window.cricketApi = (path,body) => new Promise((resolve,reject)=>{
  if(failure){reject(failure);return;}
  const id=++sequence;
  pending.set(id,{resolve,reject});
  worker.postMessage({id,path,body});
});
