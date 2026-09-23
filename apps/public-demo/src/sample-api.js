/* Public sample adapter: saved model observations, live browser application rules. */
(()=>{
 const native=window.fetch.bind(window); let samples;
 const load=()=>samples||(samples=native('/sample-data/samples.json').then(r=>{if(!r.ok)throw Error('Samples unavailable');return r.json()}));
 const statusOf=reasons=>reasons.some(r=>r.status==='decline')?'decline':reasons.some(r=>r.status==='review')?'review':'match';
 const visualLabels={pool:'a swimming pool',ocean:'the ocean',mountains:'mountains',garden:'a garden',ramp:'an entrance ramp'};
 window.fetch=async(input,options={})=>{
  const url=new URL(typeof input==='string'?input:input.url,location.href);
  if(url.origin!==location.origin||!url.pathname.startsWith('/api/'))return native(input,options);
  const s=await load();const endpoint=url.pathname.slice(5);let result;
  if(endpoint==='catalogue')result=s.catalogue;
  else if(['status','report','errors'].includes(endpoint))return native('/sample-data/'+endpoint+'.json');
  else if(endpoint==='observe')return Response.json({error:'Own-photo inference is available only in the local app.'},{status:400});
  else if(endpoint==='decide'||endpoint==='vision'){
   const body=JSON.parse(options.body||'{}'),o=s.catalogue.offers.find(o=>o.id===body.offer_id);
   if(!o)return Response.json({error:'Unknown sample'},{status:404});
   const ids=body.photo_ids||o.photos.map(p=>p.id);const original=s.vision[o.destination];
   const vision={...original,mode:'recorded',run_id:'sample-'+o.id,photos:original.photos.filter(p=>ids.includes(p.photo_id)),scope:s.scope};
   if(endpoint==='vision')result=vision;
   else{
    const receipt=s.receipts[o.id],text_profiles={},profiles={};
    for(const p of body.profiles){
     const reasons=[{task:'budget',status:o.price>p.budget?'decline':'match',text:`€${o.price} ${o.price>p.budget?'exceeds':'fits'} the €${p.budget} budget.`}];
     for(const task of p.requirements){const a=receipt.answers.find(a=>a.id===task),c=o.clauses.find(c=>c.task===task),confidence=a.probabilities[a.choice];reasons.push({task,status:a.choice===1?'decline':a.choice===2||confidence<receipt.accept_threshold?'review':'match',text:c.text,clause_id:c.id,confidence});}
     text_profiles[p.id]={status:statusOf(reasons),reasons:[...reasons]};
     for(const req of p.visual_requirements||[]){const trait=req==='avoid_steps'?'steps':req;const seen=body.include_photos?vision.photos.filter(p=>p.answers[trait].choice==='visible'&&p.answers[trait].probabilities.visible>=.8):[];reasons.push({task:'visual_'+req,status:seen.length?(req==='avoid_steps'?'decline':'match'):'review',text:seen.length?(req==='avoid_steps'?'Entrance stairs were detected in these recorded observations.':'The recorded observations show '+visualLabels[req]+'.'):(req==='avoid_steps'?'Photos cannot establish a step-free route.':'The selected photos do not clearly establish this preference.'),photo_urls:seen.map(p=>p.url)});}
     profiles[p.id]={status:statusOf(reasons),reasons};
    }
    result={...receipt,offer_id:o.id,run_id:'sample-'+o.id,mode:'recorded',scope:s.scope,profiles,text_profiles,vision:body.include_photos?{...vision,status:ids.length?'completed':'no_photos'}:{status:'disabled',photos:[],elapsed_ms:0},selected_photo_ids:body.include_photos?ids:[],elapsed_ms:receipt.text_elapsed_ms,request_profiles:body.profiles};
   }
  }else return Response.json({error:'This action requires the local app.'},{status:404});
  return Response.json(result);
 };
})();
