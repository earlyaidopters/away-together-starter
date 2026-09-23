import {test} from 'node:test';import assert from 'node:assert/strict';import {rankHolidays} from '../src/ranking.ts';
const row=(id:string,price:number,a:string,b:string)=>({offer:{id,price},receipt:{profiles:{a:{status:a},b:{status:b}}}});
test('best group fit precedes a cheaper partial match',()=>{assert.equal(rankHolidays([row('cheap',1,'match','decline'),row('fit',9,'match','match')],['a','b'])[0].offer.id,'fit')});
test('ranking uses selected travellers and price breaks equal fit',()=>{assert.equal(rankHolidays([row('a',9,'match','match'),row('b',1,'match','decline')],['a'])[0].offer.id,'b')});
test('unknown is review, never counted as a match',()=>{const r=rankHolidays([row('a',9,'review','decline')],['a','b','missing'])[0];assert.deepEqual([r.matches,r.declines,r.reviews],[0,1,2])});
test('no mutation and deterministic ties',()=>{const a=[row('z',9,'match','match'),row('a',9,'match','match')];assert.equal(rankHolidays(a,['a'])[0].offer.id,'a');assert.equal(a[0].offer.id,'z')});
