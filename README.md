# ffwrapped
## Currently offers 2023 simulationQuery service
### Description:
- Users can maximize potential of their fantasy team by simulating round-by-round selection in 10-team, 2023 NFL snake draft.
- We've run nearly 100,000 simulated fantasy drafts and combined with predictive machine learning algorithms to provide real-time estimates for which roster choices will help your team the most!

### Try it yourself!
git clone https://github.com/kevniu95/ffwrapped.git

cd ffwrapped

python3 -m venv \<yourEnvironmentName\>

source \<yourEnvironmentName\>/bin/activate

pip install -r requirements.txt

python -m src.modules.simQuery.cli

* The above steps let you run the CLI portion of the tool. Data is stored and analyzed through a Flask backend service deployed via AWS Code Runner (soon to be ECS). We need _a lot_ of data to make this analysis useful (see example below to get a feel for why) - so we'll look to do data storage via S3 or RDB (currently data is small enough to be packaged with the container itself). Check out app.py and simulationQuery.py in /src/modules/simQuery if you're interested!

### Example:
1. Oh wow, you were chosen as the 9th team in the first round of a 10-team HPPR league! I was last year too! We've run 70,000 simulated drafts with our predictive metrics. Based on this analysis, the average team picking at spot #9 in the first round can expect to score 92.6 points if they play all the right starters in a given week. (This number doesn't account for kickers and defense, so your expected max possible points on a weekly basis is even more!) Saquon looks promising this year - let's enter his id and select him
    <img width="828" alt="Screen Shot 2024-02-22 at 12 08 07 AM" src="https://github.com/kevniu95/ffwrapped/assets/86326356/590da7fd-ce91-4563-82ca-f8c5ad5d583b">

2. Not a bad choice! Looks like we can now expect to average 95.9 points a week if we play the right starters (our predictive models estimate the maximum points a team can score on a weekly basis - assuming they choose the best starters). Remember how we started with 70,000 simulated drafts? Only 6,240 of the simulated drafts resulted in Saquon Barkley going at #9. Note with each selection, the number of simulated drafts informing our current draft path goes down. The less observations there are, the less sure we are that our estimate is a good one. With that in mind, why does this "Average QB" Player have so many observations?
   
    <img width="828" alt="Screen Shot 2024-02-22 at 12 15 33 AM" src="https://github.com/kevniu95/ffwrapped/assets/86326356/5b1d916e-9b61-439a-8bd1-693526dbf55c">

4. We enter AvgQB instead of a single player's profootballreference id, and see the following: 422 simulations went down this path. That's a lot of observations relative to the other options! It's because we're looking at all simulated drafts where Saquon Barkley was taken 9th overall, and _any QB_ was taken 12th overall (for your 2nd round pick). On average, a team making these two selections can score 100 points a week if they select the best roster available to them!
   
     <img width="823" alt="Screen Shot 2024-02-22 at 12 25 13 AM" src="https://github.com/kevniu95/ffwrapped/assets/86326356/b7ebd42d-4ec5-4bb0-9133-a8e40739f8f7">

6. We're running out of observations fast! Seems like a constraint with this type of analysis! Let's say we want to group players for our next selection as well - that'll help us keep more observations backing our predicted points! We need a receiver, but we don't want just a set of _average_ wide receivers like we had for QB's. We want above-average wide-receivers! And we can get them! We just have to enter the ids of the players we want, separated by a space. If we can get our hands on any of these guys, we'll have bumped up our expected max points per week even more!
   
     <img width="816" alt="Screen Shot 2024-02-22 at 12 37 22 AM" src="https://github.com/kevniu95/ffwrapped/assets/86326356/d03d83c7-05ad-43a2-9829-111893dc033e">
