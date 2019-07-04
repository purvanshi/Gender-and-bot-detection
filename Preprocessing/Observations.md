I will share some observations here I made during the text cleaning
Note, that I do not consider the current notebook as finished, but thought that it would be helpful if I shared it anyway
Also note, that while the code processess all the tweets in the english directory, I did not check the result for each and every user, only for a select few

Issues that are still current
- For now I followed Shridhar's method for removing punctuation, but maybe in the future we can have more elaborate rules to handle fringe cases
    - For one, Programmer/Developer -> ProgrammerDeveloper
    - Also, Sometimes mark that should have spaces around don't actually have them (I know..stupid -> I knowstupid)
    - Furthemore, there are specific formats where we change the meaning by quite a lot: 6:03 (time) -> 603
- Not every # mark signifies an actual hashtag
- @ symbols can signify both twitter mentions as well as e-mail addresses
- Sometimes a tweet has nothing more than a URL, which means that the clean version is empty -> maybe we could replace the link with a special word
- Has not removed capitalization yet -> still under question whether we should do so, and if yes, to what extent
- Looks like the emoji package does not include all the emojis (missing for example: 🤦, ♂️, 3⃣0⃣ ,)

Possible features we can use later
- Manual retweets: although conventional retweets don't show here, we can still infer manual retweet, where the message starts with "RT" immediately followed by a twitter handle. My suspicion is that humans would use this more often
- Domain of link: although most links are in the twitter shortened format, if we somehow manage to infer to the domain from that, we could learn the variety of domains a twitter user links, and use this as a feature
- Emoji genders: I wonder if there is a connection between the gender of the twitter user, and the use of gendered emojis
    
Further notes:
- There is no guaranteee on the languge of tweets ("Je vous aime tous!" - in the english folder)
    

TODO: cleaning spanish text (I think maybe Pedro would be better suited for the task than I am)
