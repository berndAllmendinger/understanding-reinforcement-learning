"""Independent mathematical and game checks; no notebook training outputs required.

Run from any directory with: python tests/check_notebooks.py
The notebooks are the source of truth. We load their definitions, not fitted answers.
"""
import ast
import contextlib
import copy
import io
import json
import re
from pathlib import Path
from unittest.mock import patch
import numpy as np
import nbformat

ROOT = Path(__file__).resolve().parents[1]
COUNT = 0


def check(condition, message):
    global COUNT
    COUNT += 1
    if not condition:
        raise AssertionError(message)


def load_definitions(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            check(not any(ord(ch) < 32 and ch != '\n' for ch in cell.source),
                  'No control characters in mathematical prose')
            for expression in re.findall(r'\${1,2}([^$]+)\${1,2}', cell.source):
                check(not re.search(r'(?<![\\A-Za-z])(widehat|alpha|gamma|epsilon|theta|mathbb|mathcal|ldots|pi)\b',
                                    expression),
                      'TeX macro backslashes preserved')
    scope = {}
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        tree = ast.parse(cell.source)
        selected = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
                selected.append(node)
            elif isinstance(node, ast.Assign):
                # Uppercase configuration constants, and only those, precede tests.
                names = [n.id for n in node.targets if isinstance(n, ast.Name)]
                if names and all(n.isupper() for n in names):
                    if not any(n in {'EVALUATION_OPPONENTS', 'METHOD', 'SNAPSHOT',
                                     'PLAY_METHOD'} for n in names):
                        selected.append(node)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def close_tables(first, second):
    return first.keys() == second.keys() and all(np.allclose(first[k], second[k]) for k in first)


def common_checks(scope, sample_state):
    actions = scope['legal_actions'](sample_state)
    size = scope['ACTION_COUNT']
    rng = np.random.default_rng(55)
    # An invalid but numerically huge value must never win action selection.
    limited = 1 if size == 3 else (1, 1, 0, -1, -1, 0, 0, 0, 0)
    row = np.full(size, 999.)
    for a in scope['legal_actions'](limited):
        row[a-1] = -1.
    table = {limited: row}
    for _ in range(30):
        selected = scope['epsilon_action'](table, limited, 0.5, rng)
        check(selected in scope['legal_actions'](limited), 'Legal epsilon-greedy choice')
    probabilities = scope['policy_probabilities'](table, limited)
    legal = scope['legal_actions'](limited)
    check(np.isclose(sum(probabilities), 1), 'Policy normalised over legal actions')
    check(all(probabilities[a-1] == 0 for a in range(1,size+1) if a not in legal),
          'Illegal policy probabilities zero')
    check(scope['greedy_action']({}, sample_state) == min(actions), 'Explicit first-maximum tie')

    # Independent differentiation of the clipped objective, including flat branches.
    theta = {sample_state: np.linspace(-0.2, 0.2, size)}
    p = scope['policy_probabilities'](theta, sample_state)
    batch = [(sample_state, actions[0], 1., p[actions[0]-1]/1.05, 0.7),
             (sample_state, actions[1], -1., p[actions[1]-1]/1.4, 0.8),
             (sample_state, actions[2], -1., p[actions[2]-1]/0.7, -0.9)]
    values = {sample_state: 0.25}
    old_theta, old_values = copy.deepcopy(theta), values.copy()
    batch_copy = copy.deepcopy(batch)
    scope['ppo_epoch'](theta, values, batch, 3)

    def objective(scores):
        probs = scope['policy_probabilities']({sample_state: scores}, sample_state)
        total = 0.0
        for state, action, target, old_p, advantage in batch:
            ratio = probs[action-1] / old_p
            total += min(ratio*advantage, np.clip(ratio, 0.8, 1.2)*advantage)
        return total/3

    numeric = np.zeros(size)
    for i in range(size):
        plus, minus = old_theta[sample_state].copy(), old_theta[sample_state].copy()
        plus[i] += 1e-6
        minus[i] -= 1e-6
        numeric[i] = (objective(plus)-objective(minus))/2e-6
    actual = (theta[sample_state]-old_theta[sample_state])/scope['ACTOR_STEP']
    check(np.allclose(actual, numeric, atol=1e-8), 'PPO actor matches numerical derivative')
    expected_v = 0.25 + scope['CRITIC_STEP']*sum(row[2]-0.25 for row in batch)/3
    check(np.isclose(values[sample_state], expected_v), 'PPO critic mean squared loss derivative')
    check(batch == batch_copy, 'PPO old probabilities and advantages unchanged')
    for _ in range(3):
        scope['ppo_epoch'](theta, values, batch, 3)
    check(batch == batch_copy, 'PPO records frozen across epochs')

    # Running-mean identity and return timing do not depend on learned model output.
    records = [(sample_state, actions[0], 0., sample_state, False),
               (sample_state, actions[0], 0., sample_state, False),
               (sample_state, actions[0], 1., None, True)]
    check(scope['discounted_returns'](records) == [1.,1.,1.], 'Delayed MC returns')
    check([r[2] for r in records] == [0.,0.,1.], 'Observed rewards are not overwritten')
    check(np.isclose(0.2+0.5*((0+1*0.8)-0.2), 0.5), 'Nonterminal Q hand calculation')
    check(np.isclose(0.4+0.25*(1-0.4), 0.55), 'Terminal Q hand calculation')

    # Check the real training functions on a separately specified two-step chain.
    # The terminal reward reaches the earlier Q entry only on a later episode.
    original = {key: scope[key] for key in ['start_state','step','legal_actions','epsilon_action']}
    try:
        scope['start_state'] = lambda rng: 'first'
        scope['legal_actions'] = lambda state: [1]
        scope['epsilon_action'] = lambda *args: 1
        scope['step'] = lambda state, action, rng: (('last',0.,False) if state == 'first' else (None,1.,True))
        first_q = scope['train_q_learning'](episodes=1,seed=0)[0]
        second_q = scope['train_q_learning'](episodes=2,seed=0)[0]
        alpha = scope['ALPHA']
        check(first_q['first'][0] == 0 and np.isclose(first_q['last'][0],alpha),
              'Q first episode updates terminal reward only')
        check(np.isclose(second_q['first'][0],alpha**2), 'Q later propagates terminal evidence')
        mc = scope['train_mc'](episodes=1,seed=0)[0]
        check(mc['first'][0] == mc['last'][0] == 1, 'MC complete return trains both visited cells')
    finally:
        scope.update(original)

    # Reduced fits check repeatability and snapshot independence; no convergence assertion.
    trained = {}
    for label, function in [('Monte Carlo','train_mc'), ('SARSA','train_sarsa'),
                            ('Q-learning','train_q_learning'), ('REINFORCE','train_reinforce')]:
        fit = scope[function](episodes=150, seed=17)
        repeated = scope[function](episodes=150, seed=17)
        check(close_tables(fit[0], repeated[0]), function+' deterministic rerun')
        check(fit[1][0] == {}, function+' initial snapshot remains empty/zero')
        saved = copy.deepcopy(fit[1][150])
        key = next(iter(fit[0]))
        fit[0][key][0] += 0.123
        check(close_tables(fit[1][150], saved), function+' final snapshot independent')
        trained[label] = ('policy' if label == 'REINFORCE' else 'q', repeated[0])
    ppo = scope['train_ppo'](batches=3, batch_size=20, seed=17)
    repeated = scope['train_ppo'](batches=3, batch_size=20, seed=17)
    check(close_tables(ppo[0], repeated[0]), 'PPO deterministic rerun')
    check(ppo[2][0] == {}, 'PPO initial snapshot unchanged')
    trained['PPO'] = ('policy', ppo[0])
    scope['models'] = trained

    frozen = copy.deepcopy(trained)
    # Exercise legal input, invalid input, quitting, and a complete game for each method.
    for label in trained:
        with contextlib.redirect_stdout(io.StringIO()):
            with patch('builtins.input', return_value='q'):
                if size == 3:
                    scope['play_human'](label, matches=8, human_first=True)
                else:
                    scope['play_human'](label)
            bad_once = [True]

            def respond(prompt):
                if bad_once.pop() if bad_once else False:
                    return '99'
                labels = prompt[prompt.index('[')+1:prompt.index(']')]
                return labels.split(',')[0].strip()

            with patch('builtins.input', side_effect=respond):
                scope['play_human'](label)
        check(close_tables(trained[label][1], frozen[label][1]), 'Human play freezes '+label)

    # Independently enumerate terminal outcomes against all adversarial replies.
    for label in trained:
        if size == 3:
            def terminal_scores(n):
                a = scope['agent_action'](label,n,rng)
                left = n-a
                if left == 0:
                    return [-1]
                leaves = []
                for b in range(1,min(left,3)+1):
                    leaves.extend([1] if b == left else terminal_scores(left-b))
                return leaves
            for start in [8,9]:
                check(min(terminal_scores(start)) == scope['worst_case_score'](label,start),
                      'Nim worst-case score independently enumerated')
        else:
            def terminal_scores(board):
                action = scope['agent_action'](label,board,rng)
                after = list(board)
                after[action-1] = 1
                winner = independent_winner(tuple(after))
                if winner is not None:
                    return [winner]
                leaves = []
                for i,mark in enumerate(after):
                    if mark != 0:
                        continue
                    reply = after.copy()
                    reply[i] = -1
                    winner = independent_winner(tuple(reply))
                    leaves.extend([winner] if winner is not None else terminal_scores(tuple(reply)))
                return leaves
            check(min(terminal_scores((0,)*9)) == scope['worst_case_score'](label,(0,)*9),
                  'Tic worst-case score independently enumerated')
    return trained


def nim_checks(scope):
    for n in range(1,10):
        check(scope['legal_actions'](n) == list(range(1,min(n,3)+1)), 'Nim legal removals')
        for a in range(1,min(n,3)+1):
            left = n-a
            if left == 0:
                expected = (None,-1.,True)
            else:
                following = left-min(2,left)
                expected = (None,1.,True) if following == 0 else (following,0.,False)
            check(scope['step'](n,a,np.random.default_rng(0)) == expected, 'Nim paired transition')
        check(scope['perfect_value'](n) == (-1 if n%4 == 1 else 1), 'Nim modulo-four strategy')
    for state, action in [(1,2), (2,3), (3,0), (3,4)]:
        try:
            scope['step'](state, action, np.random.default_rng(0))
        except ValueError:
            check(True, 'Nim invalid action rejected')
        else:
            check(False, 'Nim accepted invalid action')
    common_checks(scope, 6)


def independent_winner(board):
    # Deliberately derive rows/columns/diagonals independently of the notebook list.
    lines = [board[i*3:(i+1)*3] for i in range(3)]
    lines += [board[i::3] for i in range(3)]
    lines += [(board[0],board[4],board[8]),(board[2],board[4],board[6])]
    for line in lines:
        if sum(line) == 3:
            return 1
        if sum(line) == -3:
            return -1
    return 0 if all(board) else None


def tic_checks(scope):
    frontier = {((0,)*9,1)}
    visited = set()
    x_states = 0
    while frontier:
        board, player = frontier.pop()
        if (board,player) in visited:
            continue
        visited.add((board,player))
        outcome = independent_winner(board)
        check(scope['result'](board) == outcome, 'Tic terminal status on reachable board')
        expected = [] if outcome is not None else [i+1 for i,v in enumerate(board) if v == 0]
        check(scope['legal_actions'](board) == expected, 'Tic legal mask on reachable board')
        if outcome is not None:
            continue
        if player == 1:
            x_states += 1
        for action in expected:
            updated = list(board)
            updated[action-1] = player
            updated = tuple(updated)
            check(scope['place'](board,action,player) == updated, 'Tic move and encoding')
            frontier.add((updated,-player))
    check(len(visited) == 5478, 'Known count of reachable labelled Tic-Tac-Toe boards')
    check(x_states == 2423, 'All nonterminal X-turn states counted')
    check(scope['minimax']((0,)*9,1) == 0, 'Perfect play draw')
    winning = (1,1,0,-1,-1,0,0,0,0)
    check(scope['step'](winning,3,np.random.default_rng(0)) == (None,1.,True),
          'No opponent move after immediate X win')
    for seed in range(30):
        s,r,d = scope['step']((0,)*9,5,np.random.default_rng(seed))
        check(s.count(1) == s.count(-1) == 1 and s[4] == 1 and r == 0 and not d,
              'Paired Tic transition returns to X perspective')
    common_checks(scope, (0,)*9)


if __name__ == '__main__':
    for filename, checker in [('01_nim.ipynb',nim_checks), ('02_tic_tac_toe.ipynb',tic_checks)]:
        definitions = load_definitions(ROOT/'notebooks'/filename)
        checker(definitions)
        print(filename, 'passed')
    print(COUNT, 'checks passed')
