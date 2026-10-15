| study | value | parameters | final_train_loss | final_val_loss | best_val_loss | val_ppl | train_time_sec | step_time_sec | tokens_per_sec | peak_gpu_memory_mb | attention_matrix_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| depth | 2 | 413440 | 1.4902 | 1.6841 | 1.6841 | 5.388 | 19.7 | 18.2 | 449100.4 | 142.4 |  |
| depth | 4 | 809984 | 1.3984 | 1.6036 | 1.6036 | 4.971 | 37.7 | 35.5 | 230463.3 | 249.1 |  |
| depth | 6 | 1206528 | 1.3703 | 1.5693 | 1.5693 | 4.803 | 54.7 | 51.6 | 158713.8 | 355.8 |  |
| heads | 2 | 809984 | 1.4126 | 1.6131 | 1.6131 | 5.018 | 32.9 | 30.8 | 266068.4 | 213.1 |  |
| heads | 4 | 809984 | 1.3984 | 1.6036 | 1.6036 | 4.971 | 37.3 | 35.0 | 234135.9 | 249.1 |  |
| heads | 8 | 809984 | 1.4081 | 1.61 | 1.61 | 5.003 | 46.4 | 43.9 | 186661.2 | 321.1 |  |
| context | 64 | 809984 | 1.4932 | 1.6642 | 1.6642 | 5.282 | 28.5 | 27.0 | 151586.6 | 119.5 | 8.0 |
| context | 128 | 809984 | 1.3984 | 1.6036 | 1.6036 | 4.971 | 37.0 | 34.9 | 234681.7 | 249.1 | 32.0 |
| context | 256 | 809984 | 1.3441 | 1.5632 | 1.5632 | 4.774 | 89.6 | 86.5 | 189446.4 | 613.8 | 128.0 |
| position | sinusoidal | 809984 | 1.3984 | 1.6036 | 1.6036 | 4.971 | 37.8 | 35.4 | 231678.5 | 249.1 |  |
| position | learned | 826368 | 1.4961 | 1.6884 | 1.6884 | 5.411 | 37.3 | 35.1 | 233331.1 | 249.2 |  |
| position | none | 809984 | 1.6463 | 1.816 | 1.816 | 6.147 | 37.2 | 34.9 | 234438.4 | 249.1 |  |

## Samples (prompt `ROMEO:`, temperature 0.8, seed 0)

**depth = 2**

```
ROMEO:
Say thing all at time but in they well;
Now in me not good my lord with the father!

BRUTUS:
You look Waith me shear but stops a there's dise,
In I should swear teak and bellow there had
Be to like w

```

**depth = 4**

```
ROMEO:
Say thinks a man! if God up you come,
For it well not good my lord of your battled!

Nurse:
Yes, look this the shall be common a treath.

SICINIUS:
Now, I swear this state the purge dil war
Which you

```

**depth = 6**

```
ROMEO:
Say things, bear time by us you call death?

FRIAR LAURENCE:
This controus battle so sounds grace, thou it the shall be
come to limps and in a garling sinces away.

BUCKINGHAM:
And is law and in seal

```

**heads = 2**

```
ROMEO:
Say thinks, be thou art you well time.

Nurse:
And we have my lord:
And the father! it unto gracelo, if it the shall be?
Why shall the said with of sovereign a gone.

MARCIUS:
Such is law the is seal

```

**heads = 4**

```
ROMEO:
Say thinks a man! if God up you come,
For it well not good my lord of your battled!

Nurse:
Yes, look this the shall be common a treath.

SICINIUS:
Now, I swear this state the purge dil war
Which you

```

**heads = 8**

```
ROMEO:
I am I not liege! if God, the mocks his wall be eye;
And by with us the lady had dead our sound.

GRUMIO:
Sir, sir; he commons and the makes a garlieves,
Tessave the state to drumplishly.

LUCIO:
Now

```

**context = 64**

```
ROMEO:
Say things an Clifice but thy mocking.

Nurse:
And love's such thrue in the father! is unto grace,
This impind she of thyself.

RIVERS:
Lord a garlieve in the wound land.

STOP OF
First Margan:
And w

```

**context = 128**

```
ROMEO:
Say thinks a man! if God up you come,
For it well not good my lord of your battled!

Nurse:
Yes, look this the shall be common a treath.

SICINIUS:
Now, I swear this state the purge dil war
Which you

```

**context = 256**

```
ROMEO:
Say he no general in his bidd me?

ABHORSON:
Fellow no fortune. Could your father souls,
The brant to Warwick the Cousin Warwick there's disertain.

FRIAR LAURENCE:
But and the partience.

SICINIUS:


```

**position = sinusoidal**

```
ROMEO:
Say thinks a man! if God up you come,
For it well not good my lord of your battled!

Nurse:
Yes, look this the shall be common a treath.

SICINIUS:
Now, I swear this state the purge dil war
Which you

```

**position = learned**

```
ROMEO:
Say thinks a man! is God be you come,
For aling my lord. They proud the love hadst!

Nurse:
Sir, lord; with mether of thyself.

EDWARD:
Lord a and feaver upon with the say like a gries?

BENVO:
Ay, w

```

**position = none**

```
ROMEO:
I am I no am make in a subject.

First Musighter:
Let goods wilt unto to be for sie.

HERMIONE:
How no with me sheak but the to and my say lifears
Senver up barg Nay, that the purt id
lay with lis I 

```
