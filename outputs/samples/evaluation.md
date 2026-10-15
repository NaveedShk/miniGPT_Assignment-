# Evaluation

```json
{
  "checkpoint": "/content/transformer-gpt-assignment/checkpoints/tiny_gpt.pt",
  "parameters": 809984,
  "val_loss": 1.4933,
  "val_perplexity": 4.452,
  "val_bits_per_char": 2.1544,
  "uniform_baseline_loss": 4.1744
}
```

## Fixed-prompt samples (temperature 0.8, seed 42, 300 new chars)

### Prompt 1: `'ROMEO:'`

```
ROMEO:
As look and that I make him down that noble
And they fear her place of quite, and that's then?
As it well, if uncertain famous meaning not like
colder by a present drunk and the fair both.

POMPEY:
I shall say you not a patrician on a wuld
They have done the appear secret.

DUKE OF YORK:
Not all be
```

### Prompt 2: `'JULIET:\nO Romeo,'`

```
JULIET:
O Romeo, what is the loss of thine own conjure?

CLARENCE:
My lord, do I see, prepare you for this man.

Nurse:
O perform'd is this street with your wretch,
The valiant is my lord: shall sets him herein
As my others rively of our such ause,
If hangs from his accused the meating close.

CAMILLO:
Thou art for
```

### Prompt 3: `'KING HENRY:\nWhat news'`

```
KING HENRY:
What news ling only that invocase thee?

BENVOLIO:
Who shall think you do come?

MISTRESS OVERDONE:
The projure I can not be put hunged.

First Musician:
Sirrah, nor I say, there the oaths house my lord:
If my husband, sir, that I am I not the Lady,
And that I may play her send them shall shall all tell me
H
```

### Prompt 4: `'First Citizen:\n'`

```
First Citizen:
We think the desire of so much make me my son,
That I think the subjects thou very well.
What thou wilt I see with all great call thy delivers?

ISABELLA:
The country live and remember all.

CLARENCE:
Better the prince send?

VIRGILIA:
Before the truth; my lord, I pray the liege,
Seem them him bears
```

### Prompt 5: `'To be, or not to be'`

```
To be, or not to be the prince.

SICINIUS:
Is that he proclaim'd more crown: why, is't the glock,
Whereon our soldiers best kiss than thou shouldst
All be her.

HENRY BOLINGBROKE:
I must not have straight design to the deed.

PLINCESTER:
Ay, now my lords, I would not care him friend
That I to tell thee, on a very rest
```

## Decoding strategy comparison (prompt `ROMEO:`)

| strategy | distinct words | distinct bigrams | repeated char 4-grams | real-word fraction |
|---|---|---|---|---|
| greedy (T=0) | 0.216 | 0.309 | 0.726 | 1.0 |
| T=0.5 | 0.69 | 0.99 | 0.199 | 0.98 |
| T=0.8 | 0.765 | 1.0 | 0.123 | 0.951 |
| T=1.0 | 0.765 | 0.99 | 0.115 | 0.929 |
| T=1.5 | 0.94 | 1.0 | 0.046 | 0.747 |
| T=0.8, top-k=10 | 0.77 | 1.0 | 0.133 | 0.95 |

**greedy (T=0)**

```
ROMEO:
I will not so so much a straight of the court.

LEONTES:
What is the country?

CORIOLANUS:
I will not see the prince of the country's son,
That I will be so much a straight of the sea
That the state of the seat of the seat of the sea
That the state of the state of the state of the sea
That the state of the state of the state of the sea
That the state of the state of the state of the sea
Tha
```

**T=0.5**

```
ROMEO:
I can lose thee, and there shall be a necked for thee.

PETRUCHIO:
Why, the crown that stands thee,
That would have seen to the deed hands and make him
That change he did should be stand upon his forth.

GLOUCESTER:
Then I have not the under that I did speak a sword;
The common of your brother soul, and do you say
The seen and his good friends, in his honour.

KING RICHARD III:
Go, come, fo
```

**T=0.8**

```
ROMEO:
As look and that I make him down that noble
And they fear her place of quite, and that's then?
As it well, if uncertain famous meaning not like
colder by a present drunk and the fair both.

POMPEY:
I shall say you not a patrician on a wuld
They have done the appear secret.

DUKE OF YORK:
Not all better than visit 'name I have shown.

ANTIGONUS:
So, then shall I should be both out of the day
```

**T=1.0**

```
ROMEO:
As look and for your great sun.

SICINIUS:
A fellow eyes are deliver
To dream a weeping king, from the particiu
In sigh fame, cheering noble leads, a woman's fool.

FLORIZEL:
They but not not say, but men so it not usurpose;
Now, like turn a rent mercy from a great
Meanest that consuls, when alsomething do we
were not of the pass poor clease of mother;
Or bidems that I prove to-none the day
```

**T=1.5**

```
ROMEO:
Kind Lord, days, and not the neath, annd Lodowe thought
Was go lamp'd cqure of kintnal's planly spirit.

WARWICK:
Weigh fame!--
Son, 'tis trulh, thy temp and foolish prevain,
They but not night a, but Welvesditize husband;
His heaviol'd, dark,
My more fattheral were couvertors; consuls, Thus?

LEONTES:
Come, faith's deserve! O, by to queen,
Wofuls, too hopest, Keep
Mrings o-hope, Froth.
do,
```

**T=0.8, top-k=10**

```
ROMEO:
As I have stay'd the matter'd with all,
And the words of thee that true the crown
And have bring our world.

Provost:
The mercy of the contral of his house are son?

LUCIO:
He had but not cheer and do well say 'Think your honour,
You have been it shall be my consent in this brother.

Second Senator:
No, no more comes born to be.

First Lord:
The strong hours and heart
Were the wood, the did
```
