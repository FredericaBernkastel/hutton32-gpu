	/***

Motivation: In the original von Neumann transition rules, lines of transmission states can
extend themselves by writing out binary signal trains, e.g. 10000 for extend with a right-directed
ordinary transmission state (OTS). But for construction, a dual-stranded construction arm (c-arm)
is needed, simply because the arm must be retracted after each write. I noticed that there was room
to add the needed write-and-retract operation by modifying the transition rules slightly. This
allows the machine to be greatly reduced in size and speed of replication.

Another modification was made when it was noticed that the construction could be made rotationally
invariant simply by basing the orientation of the written cell on the orientation of the one writing
it. Instead of "write an up arrow" we have "turn left". This allows us to spawn offspring in
different directions and to fill up the space with more and more copies in a manner inspired by
Langton's Loops.

A single OTS line can now act as a c-arm in any direction. Below are the signal trains:

100000 : move forward (write an OTS arrow in the same direction)
100010 : turn left
10100  : turn right
100001 : write a forward-directed OTS and retract
100011 : write a left-directed OTS and retract
10011  : write a reverse-directed OTS and retract
10101  : write a right-directed OTS and retract
101101 : write a forward-directed special transmission state (STS) and retract
110001 : write a left-directed STS and retract
110101 : write a reverse-directed STS and retract
111001 : write a right-directed STS and retract
1111   : write a confluent state and retract
101111 : retract

Achieving these features without adding new states required making some slight changes elsewhere,
though hopefully these don't affect the computation- or construction-universality of the CA. The
most important effects are listed here:

1) OTS's cannot destroy STS's. This functionality was used in von Neumann's construction and
read-write arms but isn't needed for the logic organs, as far as I know. The opposite operation
is still enabled.
2) STS lines can only construct one cell type: an OTS in the forward direction. Some logic organs
will need to be redesigned.

Under this modified JvN rule, a self-replicator can be much smaller, consisting only of a tape
contained within a repeater-emitter loop. One early example consisted of 5521 cells in total, and
replicates in 44,201 timesteps, compared with 8 billion timesteps for the smallest known JvN-32
replicator. This became possible because the construction process runs at the same speed as a moving
signal, allowing the tape to be simply stored in a repeater-emitter loop. The machine simply creates
a loop of the right size (by counting tape circuits) before allowing the tape contents to fill up
their new home.

The rotational invariance allows the machine to make multiple copies oriented in different directions.
The population growth starts off as exponential but soons slows down as the long tapes obstruct the
new copies.

Some context for these modifications to von Neumann's rule table:
Codd simplified vN's CA to a rotationally-invariant 8 states. Langton modified this to make a
self-replicating repeater-emitter, his 'loops'. Other loops were made by Sayama, Perrier, Tempesti,
Byl, Chou-Reggia, and others. So there are other CA derived from vN's that support faster replication
than that achieveable here, and some of them retain the computation- and construction-universality
that von Neumann was considering. Our modifications are mostly a historical exploration of the
possibility space around vN's CA, to explore the questions of why he made the design decisions he did.
In particular, why didn't von Neumann design for a tape loop stored within a repeater-emitter? It would
have made his machine much simpler from the beginning. Why didn't he consider write-and-retraction
instead of designing a complicated c-arm procedure? Of course this is far from criticism of vN - his
untimely death interrupted his work in this area.

Some explanation of the details of the modifications is given below:

The transition rules are as in Nobili32 (or JvN29), except the following:
1) The end of an OTS wire, when writing a new cell, adopts one of two states: excited OTS and excited
STS, standing for bits 1 and 0 respectively. After writing the cell reverts to being an OTS.
2) A sensitized cell that is about to revert to an arrow bases its direction upon that of the excited
arrow that is pointing to it.
3) A TS 'c', with a sensitized state 's' on its output that will become an OTS next (based on the
state of 'c'), reverts to the ground state if any of 'c's input is 1, else it quiesces.
4) A TS 'c', with a sensitized state 's' on its output that will become a confluent state next
(based on the state of 'c'), reverts to the first sensitized state S is any of 'c's input is one,
else it reverts to the ground state.
5) A TS 'c', with an STS on its output, reverts to the ground state if any of 'c's input is 1.

Tim Hutton <tim.hutton@gmail.com>, 2008

	***/

fn is_OTS(c: u32) -> bool {
	return c>=9u && c<=16u;
}
fn is_STS(c: u32) -> bool {
	return c>=17u && c<=24u;
}
fn is_TS(c: u32) -> bool {
	return is_OTS(c) || is_STS(c);
}
fn is_sensitized(c: u32) -> bool {
	return c>=1u && c<=8u;
}
fn is_east(c: u32) -> bool {
	return c==9u || c==13u || c==17u || c==21u;
}
fn is_north(c: u32) -> bool {
	return c==10u || c==14u || c==18u || c==22u;
}
fn is_west(c: u32) -> bool {
	return c==11u || c==15u || c==19u || c==23u;
}
fn is_south(c: u32) -> bool {
	return c==12u || c==16u || c==20u || c==24u;
}
fn is_excited(c: u32) -> bool {
	return (c>=13u && c<=16u) || (c>=21u && c<=24u);
}

fn dir(c: u32) -> u32 {	// return 0,1,2,3 encoding the direction of 'c': right,up,left,down
	return (c - 9u)%4u;
}
fn output(c: u32,n: u32,s: u32,e: u32,w: u32) -> u32 // what is the state of the cell we are pointing to?
{
	if(is_east(c)) { return e; }
	else if(is_north(c)) { return n; }
	else if(is_west(c)) { return w; }
	else if(is_south(c)) { return s; }
	else { return 0u; } // error
}
fn input(n: u32,s: u32,e: u32,w: u32) -> u32 { // what is the state of the excited cell pointing at us?
	if(is_east(w) && is_excited(w)) { return w; }
	else if(is_north(s) && is_excited(s)) { return s; }
	else if(is_west(e) && is_excited(e)) { return e; }
	else if(is_south(n) && is_excited(n)) { return n; }
	else { return 0u; } // error
}

fn output_will_become_OTS(c: u32,n: u32,s: u32,e: u32,w: u32) -> bool
{
	return output(c,n,s,e,w)==8u
		|| (output(c,n,s,e,w)==4u && is_excited(c))
		|| (output(c,n,s,e,w)==5u && !is_excited(c));
}
fn output_will_become_confluent(c: u32,n: u32,s: u32,e: u32,w: u32) -> bool
{
	return output(c,n,s,e,w)==7u && is_excited(c);
}
fn output_will_become_sensitized(c: u32,n: u32,s: u32,e: u32,w: u32) -> bool
{
	let out=output(c,n,s,e,w);
	return ((out==0u && is_excited(c)) || out==1u || out==2u || out==3u || (out==4u && !is_OTS(c)));
}
fn excited_arrow_to_us(n: u32,s: u32,e: u32,w: u32) -> bool
{
	return n==16u || n==24u || s==14u || s==22u || e==15u || e==23u || w==13u || w==21u;
}
fn excited_OTS_to_us(c: u32,n: u32,s: u32,e: u32,w: u32) -> bool { // is there an excited OTS state that will hit us next?
	return ((n==16u || n==27u || n==28u || n==30u || n==31u) && !(c==14u || c==10u))
		|| ((s==14u || s==27u || s==28u || s==30u || s==31u) && !(c==16u || c==12u))
		|| ((e==15u || e==27u || e==28u || e==29u || e==31u) && !(c==13u || c==9u))
		|| ((w==13u || w==27u || w==28u || w==29u || w==31u) && !(c==15u || c==11u));
}
fn excited_OTS_arrow_to_us(c: u32,n: u32,s: u32,e: u32,w: u32) -> bool { // is there an excited OTS arrow pointing at us?
	return (n==16u && !(c==14u || c==10u))
		|| (s==14u && !(c==16u || c==12u))
		|| (e==15u && !(c==13u || c==9u))
		|| (w==13u && !(c==15u || c==11u));
}
fn OTS_arrow_to_us(n: u32,s: u32,e: u32,w: u32) -> bool {	// is there an OTS arrow pointing at us?
	return (is_OTS(n) && is_south(n)) || (is_OTS(s) && is_north(s))
		|| (is_OTS(e) && is_west(e)) || (is_OTS(w) && is_east(w));
}
fn excited_STS_to_us(c: u32,n: u32,s: u32,e: u32,w: u32) -> bool { // is there an excited STS state that will hit us next?
	return ((n==24u || n==27u || n==28u || n==30u || n==31u) && !(c==22u || c==18u))
		|| ((s==22u || s==27u || s==28u || s==30u || s==31u) && !(c==24u || c==20u))
		|| ((e==23u || e==27u || e==28u || e==29u || e==31u) && !(c==21u || c==17u))
		|| ((w==21u || w==27u || w==28u || w==29u || w==31u) && !(c==23u || c==19u));
}
fn excited_STS_arrow_to_us(c: u32,n: u32,s: u32,e: u32,w: u32) -> bool { // is there an excited STS arrow pointing at us?
	return (n==24u && !(c==22u || c==18u))
		|| (s==22u && !(c==24u || c==20u))
		|| (e==23u && !(c==21u || c==17u))
		|| (w==21u && !(c==23u || c==19u));
}
fn all_inputs_on(n: u32,s: u32,e: u32,w: u32) -> bool {
	return (!(n==12u || s==10u || e==11u || w==9u)) && (n==16u || s==14u || e==15u || w==13u);
}
fn is_crossing(n: u32,s: u32,e: u32,w: u32) -> bool
{
	var n_inputs=0u;
	if(is_south(n)) { n_inputs++; }
	if(is_east(w)) { n_inputs++; }
	if(is_west(e)) { n_inputs++; }
	if(is_north(s)) { n_inputs++; }
	var n_outputs=0u;
	if(is_TS(n) && !is_south(n)) { n_outputs++; }
	if(is_TS(w) && !is_east(w)) { n_outputs++; }
	if(is_TS(e) && !is_west(e)) { n_outputs++; }
	if(is_TS(s) && !is_north(s)) { n_outputs++; }
	return n_inputs==2u && n_outputs==2u;
}

fn quiesce(c: u32) -> u32
{
	if(((c>=13u && c<=16u) || (c>=21u && c<=24u))) {
		return c - 4u;
	} else if(c>=26u && c<=31u) {
		return 25u;
	} else {
		return c;
	}
}
// the update function itself
fn hutton32(c: u32,n: u32,s: u32,e: u32,w: u32) -> u32
{
	if(is_OTS(c))
	{
		if(excited_STS_arrow_to_us(c,n,s,e,w)) {
			{ return 0u; }		// we get destroyed by the incoming excited STS
		} else if(excited_OTS_to_us(c,n,s,e,w)) {
			if(output_will_become_OTS(c,n,s,e,w) || (is_STS(output(c,n,s,e,w)) && !is_excited(output(c,n,s,e,w))))
				{ return 0u; }	// we become the ground state (retraction)
			else if(output_will_become_confluent(c,n,s,e,w))
				{ return 1u; }	// we become sensitized by the next input (after retraction)
			else
				{ return quiesce(c)+4u; }	// we become excited (usual OTS transmission)
		}
		else if(output_will_become_confluent(c,n,s,e,w))
			{ return 0u; }	// we become the ground state (retraction)
		else if(is_excited(c) && output_will_become_sensitized(c,n,s,e,w))
			{ return quiesce(c)+12u; }	// we become excited STS (special for end-of-wire:
					// means quiescent OTS, used to mark which cell is the sensitized cell's input)
		else
			{ return quiesce(c); }
	}
	else if(is_STS(c))
	{
		if(is_excited(c) && is_sensitized(output(c,n,s,e,w)) && OTS_arrow_to_us(n,s,e,w))
		{
			// this cell is the special mark at the end of an OTS wire, so it behaves differently
			// if output is about to finalize, we revert to ground or quiescent OTS, depending on next signal
			// if output will remain sensitized, we change to excited OTS if next signal is 1
			if(output_will_become_sensitized(c,n,s,e,w))
			{
				if(excited_OTS_arrow_to_us(c,n,s,e,w))
					{ return c - 8u; }
				else
					{ return c; }
			}
			else {
				if(excited_OTS_arrow_to_us(c,n,s,e,w))
					{return 0u;}	// write-and-retract
				else
					{return quiesce(c) - 8u;}	// revert to quiescent OTS
			}
		}
		else if(is_excited(c) && output(c,n,s,e,w)==0u){
			if(excited_STS_arrow_to_us(c,n,s,e,w))
				{return c;}	// we remain excited
			else
				{return quiesce(c);}	// we quiesce
		} else if(excited_OTS_arrow_to_us(c,n,s,e,w))
			{return 0u;}	// we get destroyed by the incoming excited OTS
		else if(excited_STS_to_us(c,n,s,e,w))
			{return quiesce(c)+4u;}	// we become excited (usual STS transmission)
		else
			 {return quiesce(c);}	// we quiesce (usual STS transmission)
	}
	else if(c==0u)
	{
		if(excited_OTS_arrow_to_us(c,n,s,e,w)) // (excludes e.g. excited confluent states)
			{return 1u;}	// we become sensitized
		else if(excited_STS_arrow_to_us(c,n,s,e,w))
			{return quiesce(input(n,s,e,w)) - 8u;}	// directly become 'forward' OTS
		else {return c;}
	}
	else if(c==1u)
	{
		if(!excited_OTS_arrow_to_us(c,n,s,e,w)) 
			{return 2u;} 	// 10
		else
			{return 3u;}	// 11
	}
	else if(c==2u)
	{
		if(!excited_OTS_arrow_to_us(c,n,s,e,w))
			{return 4u;}	// 100
		else
			{return 5u;}	// 101
	}
	else if(c==3u)
	{
		if(!excited_OTS_arrow_to_us(c,n,s,e,w))
			{return 6u;} 	// 110
		else 
			{return 7u;}	// 111
	}
	else if(c==4u)
	{
		if(!excited_OTS_arrow_to_us(c,n,s,e,w))
			{return 8u;} 	// 1000
		else
			{return ( (quiesce(input(n,s,e,w)) - 9u+2u) % 4u )+9u;}	// 1001: reverse
	}
	else if(c==5u)
	{
		if(!excited_OTS_arrow_to_us(c,n,s,e,w))
			{return ( (quiesce(input(n,s,e,w)) - 9u+3u) % 4u )+9u;} 	// 1010: turn right
		else
			{return quiesce(input(n,s,e,w))+8u;}	// 1011: STS forward
	}
	else if(c==6u)
	{
		if(!excited_OTS_arrow_to_us(c,n,s,e,w))
			{return ( (quiesce(input(n,s,e,w)) - 9u+1u) % 4u )+17u;} 	// 1100: STS turn left
		else	
			{return ( (quiesce(input(n,s,e,w)) - 9u+2u) % 4u )+17u;}	// 1101: STS reverse
	}
	else if(c==7u)
	{
		if(!excited_OTS_arrow_to_us(c,n,s,e,w))
			{return ( (quiesce(input(n,s,e,w)) - 9u+3u) % 4u )+17u;} 	// 1110: STS turn left
		else	
			{return 25u;}	// 1111
	}
	else if(c==8u)
	{
		if(!excited_OTS_arrow_to_us(c,n,s,e,w))
			{return 9u+dir(input(n,s,e,w));} 	// 10000: move forward
			//{ return 8u; }
		else
			{return 9u+dir(input(n,s,e,w)+1u);}	// 10001: turn left
	}
	else if(c==25u) 	// quiescent confluent state
	{
		if(excited_STS_arrow_to_us(c,n,s,e,w))
			{return 0u;}	// we get destroyed by the incoming excited STS
		else if(is_crossing(n,s,e,w)) // for JvN-32 crossings
		{
			if((n==16u||s==14u)&&(e==15u||w==13u))
				{return 31u;}	// double crossing
			else if(n==16u||s==14u)
				{return 30u;}	// vertical crossing
			else if(e==15u||w==13u)
				{return 29u;}	// horizontal crossing
			else
				{return 25u;}	// nothing happening
		}
		else if(all_inputs_on(n,s,e,w))
			{return 26u;}
		else
			{return 25u;}
	}
	else if(c==26u)
	{
		if(excited_STS_arrow_to_us(c,n,s,e,w))
			{return 0u;	}// we get destroyed by the incoming excited STS
		else if(all_inputs_on(n,s,e,w))
			{return 28u;}
		else
			{return 27u;}
	}
	else if(c==27u)
	{
		if(excited_STS_arrow_to_us(c,n,s,e,w))
			{return 0u;	}// we get destroyed by the incoming excited STS
		else if(all_inputs_on(n,s,e,w))
			{return 26u;}
		else
			{return 25u;}
	}
	else if(c==28u)
	{
		if(excited_STS_arrow_to_us(c,n,s,e,w))
			{return 0u;	}// we get destroyed by the incoming excited STS
		else if(all_inputs_on(n,s,e,w))
			{return 28u;}
		else
			{return 27u;}
	}
	else if(c==29u || c==30u || c==31u)
	{
		if(excited_STS_arrow_to_us(c,n,s,e,w))
			{return 0u;	}// we get destroyed by the incoming excited STS
		else if((n==16u||s==14u)&&(e==15u||w==13u))
			{return 31u;}	// double crossing
		else if(n==16u||s==14u)
			{return 30u;}	// vertical crossing
		else if(e==15u||w==13u)
			{return 29u;}	// horizontal crossing
		else
			{return 25u;}	// revert to quiescent confluent state
	}
	else
		{return c;}	// error - should be no more states
}
