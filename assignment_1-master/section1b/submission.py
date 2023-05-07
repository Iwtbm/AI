
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: notebook.ipynb

import time
from isolation import Board

# Credits if any
# 1)
# 2)
# 3)

class OpenMoveEvalFn:
    def score(self, game, my_player=None):
        """Score the current game state
        Evaluation function that outputs a score equal to how many
        moves are open for AI player on the board minus how many moves
        are open for Opponent's player on the board.

        Note:
            If you think of better evaluation function, do it in CustomEvalFn below.

            Args
                game (Board): The board and game state.
                my_player (Player object): This specifies which player you are.

            Returns:
                float: The current state's score. MyMoves-OppMoves.

            """

        # TODO: finish this function!
        s = len(game.get_player_moves(my_player)) - len(game.get_opponent_moves(my_player))

        return s
        raise NotImplementedError


######################################################################
########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
######## IF YOU WANT TO CALL OR TEST IT CREATE A NEW CELL ############
######################################################################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

class CustomEvalFn:
    def __init__(self):
        pass

    def score(self, game, my_player=None):
        """Score the current game state.

        Custom evaluation function that acts however you think it should. This
        is not required but highly encouraged if you want to build the best
        AI possible.

        Args:
            game (Board): The board and game state.
            my_player (Player object): This specifies which player you are.

        Returns:
            float: The current state's score, based on your own heuristic.
        """

        # TODO: finish this function!
        s = len(game.get_player_moves(my_player)) - len(game.get_opponent_moves(my_player))

        return s
        raise NotImplementedError

######################################################################
############ DON'T WRITE ANY CODE OUTSIDE THE CLASS! #################
######## IF YOU WANT TO CALL OR TEST IT CREATE A NEW CELL ############
######################################################################

class CustomPlayer:
    # TODO: finish this class!
    """Player that chooses a move using your evaluation function
    and a minimax algorithm with alpha-beta pruning.
    You must finish and test this player to make sure it properly
    uses minimax and alpha-beta to return a good move."""

    def __init__(self, search_depth=7, eval_fn=OpenMoveEvalFn()):
        """Initializes your player.

        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Evaluation function used by your agent
        """
        self.eval_fn = eval_fn
#         self.eval_fn = CustomEvalFn()
        self.search_depth = search_depth

    def move(self, game, time_left):
        """Called to determine one move by your agent

        Note:
            1. Do NOT change the name of this 'move' function. We are going to call
            this function directly.
            2. Call alphabeta instead of minimax once implemented.
        Args:
            game (Board): The board and game state.
            time_left (function): Used to determine time left before timeout

        Returns:
            tuple: ((int,int),(int,int),(int,int)): Your best move
        """
#         t = time.time()
#         best_move, utility = alphabeta(self, game, time_left, depth=self.search_depth)
#         print(time.time()-t)
#         t = time.time()
        d = 1
        moves_dict = {}
        while(time_left()>30):
            moves, value, moves_dict = alphabeta(self, game, time_left, depth=d, moves_dict = moves_dict)
#             utility = -10000
            if time_left()<30:
                if value > utility:
                    best_move = moves
                break

            best_move = moves
            utility = value
            d+=1
        return best_move

    def utility(self, game, my_turn):
        """You can handle special cases here (e.g. endgame)"""
        return self.eval_fn.score(game, self)



###################################################################
########## DON'T WRITE ANY CODE OUTSIDE THE CLASS! ################
###### IF YOU WANT TO CALL OR TEST IT CREATE A NEW CELL ###########
###################################################################

def minimax(player, game, time_left, depth, my_turn=True):
    """Implementation of the minimax algorithm.
    Args:
        player (CustomPlayer): This is the instantiation of CustomPlayer()
            that represents your agent. It is used to call anything you
            need from the CustomPlayer class (the utility() method, for example,
            or any class variables that belong to CustomPlayer()).
        game (Board): A board and game state.
        time_left (function): Used to determine time left before timeout
        depth: Used to track how deep you are in the search tree
        my_turn (bool): True if you are computing scores during your turn.

    Returns:
        (tuple, int): best_move, val
    """

    # TODO: finish this function!
    if my_turn:
        moves = game.get_player_moves(player)
        val = -1000
        for m in moves:
            new_game, is_over, winner = game.forecast_move(m)

            if is_over or depth == 1 or time_left()<1000:
                v = player.utility(new_game, my_turn)

            else:
                _, v = minimax(player, new_game, time_left, depth - 1, my_turn=False)

            if v>val:
                best_move = m
                val = v

    else:
        moves = game.get_opponent_moves(player)
        val = 1000

        for m in moves:
            new_game, is_over, winner = game.forecast_move(m)

            if is_over or depth == 1 or time_left()<1000:
                v = player.utility(new_game, my_turn)

            else:
                _, v = minimax(player, new_game, time_left, depth - 1, my_turn=True)
            if v<val:
                best_move = m
                val = v

#     raise NotImplementedError
    return best_move, val


######################################################################
########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
######## IF YOU WANT TO CALL OR TEST IT CREATE A NEW CELL ############
######################################################################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def alphabeta(player, game, time_left, depth, alpha=float("-inf"), beta=float("inf"), my_turn=True, moves_dict={}):
    """Implementation of the alphabeta algorithm.

    Args:
        player (CustomPlayer): This is the instantiation of CustomPlayer()
            that represents your agent. It is used to call anything you need
            from the CustomPlayer class (the utility() method, for example,
            or any class variables that belong to CustomPlayer())
        game (Board): A board and game state.
        time_left (function): Used to determine time left before timeout
        depth: Used to track how deep you are in the search tree
        alpha (float): Alpha value for pruning
        beta (float): Beta value for pruning
        my_turn (bool): True if you are computing scores during your turn.

    Returns:
        (tuple, int): best_move, val
    """
    # TODO: finish this function!
#     t = time.time()
    time_limit = 30

    if my_turn:
        moves = game.get_player_moves(player)
        val = -1000

        if len(moves_dict) != 0:
            trans_dict = {value:key for key,value in moves_dict.items() if key in moves}
            trans_moves = [trans_dict[i] for i in sorted(trans_dict, reverse = True)] + moves
            next_moves = list(set(trans_moves))
            next_moves.sort(key=trans_moves.index)
        else:
            next_moves = moves

        for m in next_moves:
            new_game, is_over, winner = game.forecast_move(m)

            if is_over:
                if new_game.__inactive_player_name__ == winner:
                    v = 500 + player.utility(new_game, my_turn)
                else:
                    v = -500 + player.utility(new_game, my_turn)

            elif depth == 1 or time_left() < time_limit:
                v = player.utility(new_game, my_turn)

            else:
                _, v, moves_dict = alphabeta(player, new_game, time_left, depth - 1, alpha, beta, my_turn=False, moves_dict=moves_dict)

            moves_dict[m] = v

            if v > val:
                best_move = m
                val = v
                alpha = max(alpha, val)

            if alpha >= beta:
                break

    else:
        moves = game.get_opponent_moves(player)
        val = 1000

        if len(moves_dict) != 0:
            trans_dict = {value:key for key,value in moves_dict.items() if key in moves}
            trans_moves = [trans_dict[i] for i in sorted(trans_dict)] + moves
            next_moves = list(set(trans_moves))
            next_moves.sort(key=trans_moves.index)
        else:
            next_moves = moves

        for m in next_moves:
            new_game, is_over, winner = game.forecast_move(m)

            if is_over:
                if new_game.__inactive_player_name__ == winner:
                    v = -500 + player.utility(new_game, my_turn)
                else:
                    v = 500 + player.utility(new_game, my_turn)

            elif depth == 1 or time_left()<time_limit:
                v = player.utility(new_game, my_turn)

            else:
                _, v, moves_dict = alphabeta(player, new_game, time_left, depth - 1, alpha, beta, my_turn=True, moves_dict=moves_dict)

            moves_dict[m] = v

            if v < val:
                best_move = m
                val = v

                beta = min(beta, val)

            if alpha >= beta:
                break

#     raise NotImplementedError
    return best_move, val, moves_dict


######################################################################
########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
######## IF YOU WANT TO CALL OR TEST IT CREATE A NEW CELL ############
######################################################################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
# tests.name_of_the_test #you can uncomment this line to run your test
# tests.beatRandom(CustomPlayer)
################ END OF LOCAL TEST CODE SECTION ######################