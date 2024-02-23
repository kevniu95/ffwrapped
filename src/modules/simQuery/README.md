# ffsimquery service
Users can maximize the potential of their fantasy team by simulating round-by-round selections in a 10-team 2023 NFL Snake Draft!

We've run nearly 100,000 (for now) simulated fantasy drafts and combined the results with predictive machine learning algorithms to provide estimates for which roster choices will help your team the most!

# Download the tool
Before you begin, make sure you have the following installed!

- Python 3.9
- pip
- git

## Installation steps

1.  **Clone the repository**  
    ```
    git clone https://github.com/kevniu95/ffwrapped.git
    ```
    Navigate to the project directory
    ```
    cd ffwrapped
    ```
2.  **Set up a virtual environment**
    ```
    python3 -m venv ffwrapped_env
    ```
    Activate the virtual environment

    - **Windows**
      ```
      ffwrapped_env\Scripts\activate
      ```
    - **Mac**
      ```
      source ffwrapped_env/bin/activate
      ```
3.  **Install dependencies**
      ```
      pip install -r requirements.txt
      ```
4.  **Run the application**
      ```
      python -m src.modules.simQuery.cli
      ```
    Be sure to execute this from the root of project directory ```path/to/ffwrapped```


# Example
Oh wow, you were chosen as the 9th team in the first round of a 10-team HPPR league- I was last year too! Let's see how we can use the ffsimquery tool to help us out!

## 1. Get started

Enter "9" when asked which team you'd like to draft for

<img width="457" alt="Screen Shot 2024-02-22 at 8 51 03 PM" src="https://github.com/kevniu95/ffwrapped/assets/86326356/460e7ff7-64aa-4161-9c71-2ef874f6ebfd">

## 2. Reading data summary

Wow, look at all that data! What does it mean?

### Round Information
The "Round 1 Info" section tells us where we are in the draft, and gives some info on previous selections. The important data points here are: 

<img width="825" alt="Screen Shot 2024-02-22 at 9 18 12 PM" src="https://github.com/kevniu95/ffwrapped/assets/86326356/947f15b5-38ba-493a-aedd-58b8d687cc10">

-**"Available samples..."**
  
  - How many simulated leagues (we have about 70,000 in total) had a team in the exact same position as you! This means they are at the same pick and have made the same selections so far.
  - You haven't made any picks! So every league - all 70,000 of them - have had a team in your situation: team 9 picking with 9th pick of 1st round and no selections made yet.

-**"Expected max points..."**

  - For all the teams that have been in this position, we calculated their "expected weekly best score"
  - What is "expected weekly best score"?

     - Our machine learning algorithms have simulated thousands of fantasy leagues on 7 years of NFL data. For each simulated team, we calculate the best team of starters that could be put together in any given week - then we average that number over all weeks in the season. This is the value we're predicting.
       - In short we're saying: on any given week, how many points will my best starting lineup produce?
       - Note: this metric does not include kickers and defense, so you can add 10-20 points to this number.
     - Why do it like this? More to come on this later. The short version, is that fantasy football is a composite of two skills

       - Drafting and
       - Selecting starters from drafted team

        By assuming every fantasy owner can always select a perfect starting lineup (2), we are in, some sense, "isolating" drafting ability as the only variable for evaluation - a desirable outcome for a draft tool!

What do we learn from expected max points at this point, when we haven't drafted anyone? We haven't chosen anyone, so this is the metaphorical starting line. On average, teams that draft 9th in the 1st round score have "expected weekly max points" of 92.6.

### Player Table
Great, now let's focus on the big table of players below that round information...

<img width="825" alt="Screen Shot 2024-02-22 at 9 18 12 PM" src="https://github.com/kevniu95/ffwrapped/assets/86326356/4859e304-917a-4719-abc4-9b5f607e2f40">

For all the simulated leagues where a team was in the same position as you, this is the set of players that were chosen. Let's say we're pretty interested in Saquon Barkley with the 9th pick. If we go to his row, we see that

- Of the 70,197 simulated leagues that were in our current position, 6,240 of them chose Barkley.
- On average teams that drafted Barkley 9th overall had an "expected max points" score of 95.93. Some had higher expected max points, and some had lower, depending on who they picked afterwards, of course. Remember, that before we picked Barkley, the average team selecting 9th overall had an expected max score of 92.6. So it's a great pick!

## 3. Make pick
Let's go ahead and lock in that pick! We see in Barkley's row, there is an identifier. Let's type ```BarkSa00``` in and hit "enter"...

<img width="744" alt="Screen Shot 2024-02-22 at 9 30 32 PM" src="https://github.com/kevniu95/ffwrapped/assets/86326356/74305453-4854-4708-ac00-2b23c44a8d5f">

## 4. Read Round 2 Data
So what happens now? Well, we do it all over again! Let's check out our new round summary, updated for Round 2! 

### Round Information
Do some of those numbers look familiar?

<img width="825" alt="Screen Shot 2024-02-22 at 9 31 13 PM" src="https://github.com/kevniu95/ffwrapped/assets/86326356/538e1b70-9e5e-4209-ad05-5d6d1d765efe">

Sure they do! Now that we've chosen Barkley, our summary shows that

 - There are 6,240 leagues where the 9th overall selection was Barkley.
 - On average teams thath selected Barkley 9th had expected max points score of 95.9.
 
Of course, we saw  those figures from his row back in the Player Table for round 1!

### Player Table
Now let's check out the player table.

<img width="825" alt="Screen Shot 2024-02-22 at 9 31 13 PM" src="https://github.com/kevniu95/ffwrapped/assets/86326356/2548d29b-86ef-42b1-b230-7481106fed12">

#### On counts, a limitation of this analysis
The counts went down fast! If we choose Patrick Mahomes, only 253 of the original 70,000 simulated drafts will match our current draft path (i.e., they had Barkley going 9th overall to your team, then Mahomes going 12th overall to your team). That's a pretty serious limitation to this type of analysis - it needs a lot of data!

So we can explore less common draft paths, but that will make us less sure of our estimates going forward. Since all subsequent draft paths are a subset of this current draft path, the number will keep getting smaller, whoever you choose!

### Aggregated selections

#### By position
Who is this "QB - Average" guy at row-index 8? 

In part, he's a solution to the count problem. When we select this guy (by entering id ```AvgQB``` as the id), we are saying: show me all draft paths where 

- Saquon Barkley is taken 9th overall, and then
- Any QB is taken 12th overall.

Now our analysis is more robust, we have more data backing our draft path if we pick "AvgQB" (422 observations) over a specific QB like Mahomes (253 observations).

#### By specific player
But maybe you don't want just any average QB. We want the best of the best - say, either Patrick Mahomes or Josh Allen. Just enter both their ids, separated by a space. 

```MahoPa00 AlleJo02```

If we do this we will be shown all draft paths where 

- Saquon is taken 9th overall, and then 
- Either Patrick Mahomes or Josh Allen is taken 12th overall.

You can do this with any number of players, of any position!

#### Aggregating tradeoffs
So aggregating is in part, a solution to the limited data problem. The tradeoff, though, is it will make our analysis less specific. Seems like a bit of a cop-out, no?

Well aggregating players like this can be a useful tool too! For instance, if you're picking 9th overall, should you choose a RB or a WR? Does that change if you're picking 1st overall? Now we can have an opinionated stance on those types of questions.

## 5. goback and position-based analysis
Let's dig further into this question of position-based drafting. In fact, let's go back to the first round and look at this more closely. Just type "goback" and hit enter.

<img width="813" alt="Screen Shot 2024-02-22 at 9 54 34 PM" src="https://github.com/kevniu95/ffwrapped/assets/86326356/01dcd5d8-fbdc-46ad-9042-52ca5fafbe59">

Great! Back in round 1, pick 9 - where we started. 

### A surprising finding: first-round QB?
Let's focus specifically on the position averages. What we see here is a bit of a counterintuitive take from our models. They suggest that on average, if you take any QB at 9, that's a better move for your expected max points than taking a RB at 9. And taking either a QB or RB is a much better move than taking a WR.

That runs counter to conventional wisdom for sure - Patrick Mahomes, the predicted top QB going into the year was being selected 16th off the board on average. Is there merit to the numbers we're seeing here?

### Keep draft context in mind
Perhaps there is - but remember to use this tool in context. Yes, the simquery tool says selecting AvgQB at #9 will give you more expected max points than selecting AvgRB. But Patrick Mahomes accounts for 1,451 of those draft paths, and Josh Allen accounts for 617 (over 80% of the 2,368 observations backing AvgQB!). And their average draft positions are 16.5 and 20.

If you're picking 9 now, that means your next pick is 12 overall. Mahomes and Allen will almost certainly be on the board for your round 2 pick!

So even if AvgQB looks like a better pick than AvgRB with your first overall pick - maybe the real question we should be asking ourselves, is, which of these two options is better:

- Choose QB round 1, then RB round 2
  
  **OR**
- Choose RB round 1, then QB round 2

You have access to the simquery tool, you know how to 'goback' and you know how to select players or position averages. Try to find the answer!

## 6. Closing thoughts
Hopefully, it's clear with the simple, rhetorical example from above that there's a lot of potential paths to be explored. If you greedily (describing the algorithm, not your qualities as a person) choose the highest-average player or position group from simquery in each successive round - that might not be the best strategy overall.

There are many complex paths and hypotheses to test - but now you've got a tool to help! Happy analyzing!
