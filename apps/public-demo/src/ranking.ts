type Entry={offer:{id:string;price:number};receipt:{profiles:Record<string,{status:string}>}};
export function rankHolidays<T extends Entry>(entries:T[],ids:string[]){
 return entries.map(entry=>({...entry,matches:ids.filter(id=>entry.receipt.profiles[id]?.status==='match').length,declines:ids.filter(id=>entry.receipt.profiles[id]?.status==='decline').length,reviews:ids.filter(id=>!['match','decline'].includes(entry.receipt.profiles[id]?.status)).length}))
 .sort((a,b)=>b.matches-a.matches||a.declines-b.declines||a.offer.price-b.offer.price||a.offer.id.localeCompare(b.offer.id));
}
