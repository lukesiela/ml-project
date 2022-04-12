import argparse
import sys
import pandas as pd
from tqdm import tqdm

# usage 
def usage():
    print('Usage: ./preprocessor.py --data <data_filepath> --cards <carddictionary_filepath> --output <newdata_filepath>')
    print('Note: all arguments must be present to run.')
    exit(1)

# parse arguments
def parse():
    if len(sys.argv) != 7:
        usage()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data')
    parser.add_argument('--cards', dest='cards')
    parser.add_argument('--output', dest='output')
    return parser.parse_args()

# return card dictionary given card master list file from kaggle
def get_cardlist(cardlistfile):
    return {row[1][0]:row[1][1] for row in pd.read_csv(cardlistfile).iterrows()}

# translate data to numerical only while preserving information
def main():
    args = parse()
    
    # read files
    matchups = pd.read_csv(args.data)
    cards = get_cardlist(args.cards)

    # create new dataframe with translated feature names
    new_feats = ['p1_' + str(card_id) for card_id in cards.keys()]
    new_feats.extend(['p2_' + str(card_id) for card_id in cards.keys()])
    new_feats.append('label')

    # subsetting dataframes with only deck information
    deck_matchups = matchups[['winner.card1.id','winner.card1.level','winner.card2.id','winner.card2.level','winner.card3.id','winner.card3.level','winner.card4.id','winner.card4.level','winner.card5.id','winner.card5.level','winner.card6.id','winner.card6.level','winner.card7.id','winner.card7.level','winner.card8.id','winner.card8.level','winner.elixir.average','loser.card1.id','loser.card1.level','loser.card2.id','loser.card2.level','loser.card3.id','loser.card3.level','loser.card4.id','loser.card4.level','loser.card5.id','loser.card5.level','loser.card6.id','loser.card6.level','loser.card7.id','loser.card7.level','loser.card8.id','loser.card8.level','loser.elixir.average']]
    card_matchups = matchups[['winner.card1.id','winner.card2.id','winner.card3.id','winner.card4.id','winner.card5.id','winner.card6.id','winner.card7.id','winner.card8.id','loser.card1.id','loser.card2.id','loser.card3.id','loser.card4.id','loser.card5.id','loser.card6.id','loser.card7.id','loser.card8.id']]
    level_matchups = matchups[['winner.card1.level','winner.card2.level','winner.card3.level','winner.card4.level','winner.card5.level','winner.card6.level','winner.card7.level','winner.card8.level','loser.card1.level','loser.card2.level','loser.card3.level','loser.card4.level','loser.card5.level','loser.card6.level','loser.card7.level','loser.card8.level']]

    even = True
    first = True

    # translating data
    for mi, card_matchup in tqdm(card_matchups.iterrows()):
        new_data = pd.DataFrame(columns=new_feats)
        newrow = dict()

        # loop through cards
        for fi, feature in enumerate(card_matchup):
            if (fi < 8 and even) or (fi >= 8 and not even):
                newrow['p1_' + str(feature)] = level_matchups.iat[mi,fi]
            else:
                newrow['p2_' + str(feature)] = level_matchups.iat[mi,fi]

        # fill columns with 0s, add label
        new_data = pd.concat([new_data, pd.DataFrame(newrow, index=[mi])]).fillna(0).astype(int)
        new_data['label'] = 1 if even else 0
        even = not even

        # head the output file
        if first:
            new_data.to_csv(args.output, mode='a')
            first = not first
        else:
            new_data.to_csv(args.output, mode='a', header=False)

if __name__ == "__main__":
    main()