Operator note 21:27: since the reward change the learned arm is not better than the scripted one yet (rim contacts,
episodes 18-24: learned 2,1,2,3,2 vs scripted 2,2,2; steps 18,15,18,19,17 vs 18,18,18). Please diagnose in your next
turn, briefly: does the actor's first-descent mean move toward the entry offsets the touch search found in past
episodes (touch.best)? If not, the data say exactly what the first action should have been: add hindsight
relabelling for the first descent (the found entry offset as an extra transition with the return of a 0-contact
entry), or raise the advantage weight on first-descent transitions. This is learning from the robot's own experience
(the scripted arm stays constant), so it is a fair learned-vs-scripted comparison. No change to the scripted arm.
