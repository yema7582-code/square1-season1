# Target leak in "Call the game" (found 2026-10-21, NOT used)

Every row ships each team's pre-game Elo and its game number in the season. FiveThirtyEight's pre-game Elo for a team's next game equals its post-game Elo for this game, and that update is driven by the margin of victory. So if you sort a team's games by `home_seasongame`/`away_seasongame` within a season and take next-game Elo minus this-game Elo, you largely recover the withheld score margin.

On our stand-in (last 3 train seasons): corr(next-Elo delta, margin) = **0.82**, covering 98.8% of games. A one-feature median regression on it gives MAE **5.4** (in-sample, for illustration only), against 9.25 for our honest model and 9.47 for the organisers' default GBT. The data card says post-game Elo was withheld, so this is clearly against the intent of the rules ("never the withheld targets from any source").

Decision: model.py uses only past games (Elo momentum over the last 5/10 games and since the opener). Nothing looks forward.
Status: reported to the organisers by email so they can fix the file.
