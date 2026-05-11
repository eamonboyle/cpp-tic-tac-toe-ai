// Beginning C++ Game Programming — Tic-Tac-Toe with optional self-play Q-learning AI.
// The learned agent trains without human labels: only win/loss/tie outcomes shape Q-values.

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

// Piece markers (globals retained from original tutorial layout)
const char X = 'X';
const char O = 'O';
const char EMPTY = ' ';
const char TIE = 'T';
const char NO_ONE = 'N';

using QTable = unordered_map<string, array<double, 9>>;

void instructions();
char askYesNo(const string& question);
int askNumber(const string& question, int high, int low = 0);

char humanPiece();
char opponent(char piece);

void displayBoard(const vector<char>& board);
char winner(const vector<char>& board);

bool isLegal(const vector<char>& board, int move);
int humanMove(const vector<char>& board, char /*human*/);
int computerMoveHeuristic(vector<char> board, char computer);
int computerMoveQLearning(vector<char> board, char /*computer*/, QTable& qTable);

void announceWinner(char result, char computer, char human);

string boardKey(const vector<char>& board);
char sideToMove(const vector<char>& board);
array<double, 9>& ensureQRow(QTable& q, const string& key);
double maxLegalQ(const string& state, const vector<char>& board, const QTable& q);
int pickEpsilonGreedy(const vector<char>& board, const string& state, QTable& q,
	double epsilon, mt19937& rng);
void trainSelfPlay(QTable& q, int episodes, double alpha, double gamma, mt19937& rng);

bool saveQTable(const string& path, const QTable& q);
bool loadQTable(const string& path, QTable& q);

void playVersusComputer(bool useLearned, QTable& q);

int main()
{
	random_device rd;
	mt19937 rng(rd());

	QTable qTable;
	const string defaultSavePath = "ttt_qlearn.txt";

	for (;;)
	{
		cout << "\n=== Tic-Tac-Toe ===\n";
		cout << "0) Quit\n";
		cout << "1) Play vs heuristic AI (original rule-based)\n";
		cout << "2) Play vs Q-learning AI (train first if new)\n";
		cout << "3) Train Q-learning AI with self-play (no human labels)\n";
		cout << "4) Save learned Q-table to file\n";
		cout << "5) Load Q-table from file\n";

		int choice = askNumber("Menu choice", 5, 0);
		if (choice == 0)
			break;

		switch (choice)
		{
		case 1:
			playVersusComputer(false, qTable);
			break;
		case 2:
			if (qTable.empty())
			{
				if (askYesNo("\nNo Q-table in memory. Train now?") == 'y')
				{
					int ep = askNumber("Self-play episodes", 500000, 1000);
					auto t0 = chrono::steady_clock::now();
					trainSelfPlay(qTable, ep, 0.2, 1.0, rng);
					auto ms = chrono::duration_cast<chrono::milliseconds>(
						chrono::steady_clock::now() - t0).count();
					cout << "Training finished in " << ms << " ms; states learned: "
						<< qTable.size() << "\n";
				}
			}
			playVersusComputer(true, qTable);
			break;
		case 3:
		{
			int ep = askNumber("Self-play episodes", 1000000, 1000);
			auto t0 = chrono::steady_clock::now();
			trainSelfPlay(qTable, ep, 0.2, 1.0, rng);
			auto ms = chrono::duration_cast<chrono::milliseconds>(
				chrono::steady_clock::now() - t0).count();
			cout << "Done in " << ms << " ms; distinct states: " << qTable.size() << "\n";
			ostringstream savePrompt;
			savePrompt << "Save to \"" << defaultSavePath << "\"?";
			if (askYesNo(savePrompt.str()) == 'y')
			{
				if (saveQTable(defaultSavePath, qTable))
					cout << "Saved " << qTable.size() << " states.\n";
				else
					cout << "Save failed.\n";
			}
			break;
		}
		case 4:
		{
			cout << "File path [" << defaultSavePath << "]: ";
			string path;
			cin.ignore();
			getline(cin, path);
			if (path.empty())
				path = defaultSavePath;
			if (saveQTable(path, qTable))
				cout << "Saved " << qTable.size() << " states.\n";
			else
				cout << "Save failed.\n";
			break;
		}
		case 5:
		{
			cout << "File path [" << defaultSavePath << "]: ";
			string path;
			cin.ignore();
			getline(cin, path);
			if (path.empty())
				path = defaultSavePath;
			if (loadQTable(path, qTable))
				cout << "Loaded " << qTable.size() << " states.\n";
			else
				cout << "Load failed.\n";
			break;
		}
		default:
			break;
		}
	}

	return 0;
}

void instructions()
{
	cout << "Welcome to the ultimate man-machine showdown: Tic-Tac-Toe.\n";
	cout << " -- where human brain is pit against silicon processor\n\n";

	cout << "Make your move known by entering a number, 0-8. The number\n";
	cout << "corresponds to the desired board position, as illustrated:\n\n";

	cout << "     0 | 1 | 2\n";
	cout << "     ---------\n";
	cout << "     3 | 4 | 5\n";
	cout << "     ---------\n";
	cout << "     6 | 7 | 8\n\n";

	cout << "Prepare yourself, human. The battle is about to begin.\n\n";
}

char askYesNo(const string& question)
{
	char response = ' ';

	do
	{
		if (!question.empty())
			cout << question;
		cout << " (y/n): ";
		cin >> response;
	} while (response != 'y' && response != 'n');

	return response;
}

int askNumber(const string& question, int high, int low)
{
	int number = low - 1;

	do
	{
		cout << question << " (" << low << "-" << high << "): ";
		cin >> number;
	} while (number > high || number < low);

	return number;
}

char humanPiece()
{
	char goFirst = askYesNo("Do you require the first move?");

	if (goFirst == 'y')
	{
		cout << "\nThen take the first move. You will need it.\n";
		return X;
	}

	cout << "\nYour bravery will be your undoing... I will go first.\n";
	return O;
}

char opponent(char piece)
{
	if (piece == X)
		return O;
	return X;
}

void displayBoard(const vector<char>& board)
{
	cout << "\n-----------------------------------------------------\n";
	cout << "\n\t" << board[0] << " | " << board[1] << " | " << board[2];
	cout << "\n\t" << "---------";
	cout << "\n\t" << board[3] << " | " << board[4] << " | " << board[5];
	cout << "\n\t" << "---------";
	cout << "\n\t" << board[6] << " | " << board[7] << " | " << board[8] << "         ";
}

char winner(const vector<char>& board)
{
	const int WINNING_ROWS[8][3] = {
		{0, 1, 2},
		{3, 4, 5},
		{6, 7, 8},
		{0, 3, 6},
		{1, 4, 7},
		{2, 5, 8},
		{0, 4, 8},
		{2, 4, 6}
	};

	const int TOTAL_ROWS = 8;

	for (int row = 0; row < TOTAL_ROWS; ++row)
	{
		if ((board[WINNING_ROWS[row][0]] != EMPTY) &&
			(board[WINNING_ROWS[row][0]] == board[WINNING_ROWS[row][1]]) &&
			(board[WINNING_ROWS[row][1]] == board[WINNING_ROWS[row][2]]))
		{
			return board[WINNING_ROWS[row][0]];
		}
	}

	if (count(board.begin(), board.end(), EMPTY) == 0)
		return TIE;

	return NO_ONE;
}

bool isLegal(const vector<char>& board, int move)
{
	return move >= 0 && move < static_cast<int>(board.size()) && board[move] == EMPTY;
}

int humanMove(const vector<char>& board, char /*human*/)
{
	int move = askNumber("Where will you move?", static_cast<int>(board.size()) - 1);

	while (!isLegal(board, move))
	{
		cout << "\nThat square is already occupied, foolish human.\n";
		move = askNumber("Where will you move?", static_cast<int>(board.size()) - 1);
	}

	cout << "Fine...\n";

	return move;
}

int computerMoveHeuristic(vector<char> board, char computer)
{
	cout << "I shall take square number ";

	for (size_t move = 0; move < board.size(); ++move)
	{
		if (isLegal(board, static_cast<int>(move)))
		{
			board[move] = computer;

			if (winner(board) == computer)
			{
				cout << move << endl;
				return static_cast<int>(move);
			}

			board[move] = EMPTY;
		}
	}

	char human = opponent(computer);

	for (size_t move = 0; move < board.size(); ++move)
	{
		if (isLegal(board, static_cast<int>(move)))
		{
			board[move] = human;

			if (winner(board) == human)
			{
				cout << move << endl;
				return static_cast<int>(move);
			}

			board[move] = EMPTY;
		}
	}

	const int BEST_MOVES[] = { 4, 0, 2, 6, 8, 1, 3, 5, 7 };

	for (int i = 0; i < 9; ++i)
	{
		int move = BEST_MOVES[i];

		if (isLegal(board, move))
		{
			cout << move << endl;
			return move;
		}
	}

	return 0;
}

void announceWinner(char result, char computer, char human)
{
	if (result == computer)
	{
		cout << computer << "'s won!\n";
		cout << "\n\nAs I predicted, human, I am triumphant once more -- proof\n";
		cout << "that computers are superior to humans in all regards.\n";
	}
	else if (result == human)
	{
		cout << human << "'s won!\n";
		cout << "\n\nNo, no! It cannot be! Somehow you tricked me, human.\n";
		cout << "But never again! I, the computer, so swear it!\n";
	}
	else
	{
		cout << "It's a tie.\n";
		cout << "\n\nYou were most lucky, human, and somehow managed to tie me.\n";
		cout << "Celebrate... for this is the best you will ever achieve.\n";
	}
}

string boardKey(const vector<char>& board)
{
	string s;
	s.reserve(board.size());
	for (char c : board)
		s += c;
	return s;
}

char sideToMove(const vector<char>& board)
{
	int nx = static_cast<int>(count(board.begin(), board.end(), X));
	int no = static_cast<int>(count(board.begin(), board.end(), O));
	return (nx == no) ? X : O;
}

array<double, 9>& ensureQRow(QTable& q, const string& key)
{
	auto [it, inserted] = q.try_emplace(key);
	if (inserted)
		it->second.fill(0.0);
	return it->second;
}

double maxLegalQ(const string& state, const vector<char>& board, const QTable& q)
{
	auto it = q.find(state);
	double best = -1e100;
	bool any = false;

	for (int a = 0; a < 9; ++a)
	{
		if (!isLegal(board, a))
			continue;
		double val = (it != q.end()) ? it->second[static_cast<size_t>(a)] : 0.0;
		if (!any || val > best)
		{
			best = val;
			any = true;
		}
	}

	return any ? best : 0.0;
}

int pickEpsilonGreedy(const vector<char>& board, const string& state, QTable& q,
	double epsilon, mt19937& rng)
{
	vector<int> legal;
	for (int i = 0; i < 9; ++i)
		if (isLegal(board, i))
			legal.push_back(i);

	uniform_real_distribution<double> uni01(0.0, 1.0);
	if (uni01(rng) < epsilon)
	{
		uniform_int_distribution<size_t> pick(0, legal.size() - 1);
		return legal[pick(rng)];
	}

	array<double, 9>& row = ensureQRow(q, state);
	double best = -1e100;
	vector<int> bestMoves;

	for (int a : legal)
	{
		double v = row[static_cast<size_t>(a)];
		if (v > best + 1e-12)
		{
			best = v;
			bestMoves.clear();
			bestMoves.push_back(a);
		}
		else if (abs(v - best) <= 1e-12)
			bestMoves.push_back(a);
	}

	uniform_int_distribution<size_t> tie(0, bestMoves.size() - 1);
	return bestMoves[tie(rng)];
}

void trainSelfPlay(QTable& q, int episodes, double alpha, double gamma, mt19937& rng)
{
	uniform_real_distribution<double> uni01(0.0, 1.0);

	for (int ep = 0; ep < episodes; ++ep)
	{
		double epsilon = 0.35 * (1.0 - static_cast<double>(ep) / max(episodes, 1)) + 0.05;
		vector<char> board(9, EMPTY);

		while (winner(board) == NO_ONE)
		{
			string state = boardKey(board);
			char player = sideToMove(board);
			int action = pickEpsilonGreedy(board, state, q, epsilon, rng);

			array<double, 9>& row = ensureQRow(q, state);
			board[static_cast<size_t>(action)] = player;

			char w = winner(board);
			if (w != NO_ONE)
			{
				double R = 0.0;
				if (w == TIE)
					R = 0.0;
				else if (w == player)
					R = 1.0;
				else
					R = -1.0;

				row[static_cast<size_t>(action)] +=
					alpha * (R - row[static_cast<size_t>(action)]);
				break;
			}

			string nextState = boardKey(board);
			double maxOpp = maxLegalQ(nextState, board, q);
			row[static_cast<size_t>(action)] +=
				alpha * (gamma * (-maxOpp) - row[static_cast<size_t>(action)]);
		}

		if ((ep + 1) % max(episodes / 10, 1) == 0 || ep + 1 == episodes)
			cout << "Episode " << (ep + 1) << " / " << episodes << "\r" << flush;
	}

	cout << "\n";
}

int computerMoveQLearning(vector<char> board, char /*computer*/, QTable& qTable)
{
	cout << "I shall take square number ";

	string state = boardKey(board);
	vector<int> legal;
	for (int i = 0; i < 9; ++i)
		if (isLegal(board, i))
			legal.push_back(i);

	auto it = qTable.find(state);
	if (it == qTable.end())
	{
		mt19937 rng(random_device{}());
		uniform_int_distribution<size_t> pick(0, legal.size() - 1);
		int m = legal[pick(rng)];
		cout << m << " (random — state unseen)\n";
		return m;
	}

	double best = -1e100;
	vector<int> bestMoves;
	for (int a : legal)
	{
		double v = it->second[static_cast<size_t>(a)];
		if (v > best + 1e-12)
		{
			best = v;
			bestMoves.clear();
			bestMoves.push_back(a);
		}
		else if (abs(v - best) <= 1e-12)
			bestMoves.push_back(a);
	}

	mt19937 rng(random_device{}());
	uniform_int_distribution<size_t> tie(0, bestMoves.size() - 1);
	int choice = bestMoves[tie(rng)];
	cout << choice << endl;
	return choice;
}

static string boardKeyToToken(const string& key)
{
	string t = key;
	for (char& c : t)
	{
		if (c == EMPTY)
			c = '.';
	}
	return t;
}

static string tokenToBoardKey(const string& t)
{
	if (t.size() != 9)
		return string();

	string key = t;
	for (char& c : key)
	{
		if (c == '.')
			c = EMPTY;
	}
	return key;
}

bool saveQTable(const string& path, const QTable& q)
{
	ofstream out(path);
	if (!out)
		return false;

	out << q.size() << "\n";
	for (const auto& kv : q)
	{
		out << boardKeyToToken(kv.first);
		for (double v : kv.second)
			out << " " << fixed << setprecision(6) << v;
		out << "\n";
	}

	return static_cast<bool>(out);
}

bool loadQTable(const string& path, QTable& q)
{
	ifstream in(path);
	if (!in)
		return false;

	size_t n = 0;
	in >> n;
	string line;
	getline(in, line);

	q.clear();
	for (size_t i = 0; i < n; ++i)
	{
		if (!getline(in, line))
			return false;

		istringstream iss(line);
		string tok;
		iss >> tok;
		string key9 = tokenToBoardKey(tok);
		if (key9.size() != 9)
			return false;

		array<double, 9> row{};
		for (int j = 0; j < 9; ++j)
		{
			if (!(iss >> row[static_cast<size_t>(j)]))
				return false;
		}
		q[key9] = row;
	}

	return true;
}

void playVersusComputer(bool useLearned, QTable& q)
{
	const int NUM_SQUARES = 9;
	vector<char> board(NUM_SQUARES, EMPTY);

	instructions();
	char human = humanPiece();
	char computer = opponent(human);
	char turn = X;
	displayBoard(board);

	while (winner(board) == NO_ONE)
	{
		int move = 0;
		if (turn == human)
		{
			move = humanMove(board, human);
			board[static_cast<size_t>(move)] = human;
		}
		else
		{
			if (useLearned)
				move = computerMoveQLearning(board, computer, q);
			else
				move = computerMoveHeuristic(board, computer);
			board[static_cast<size_t>(move)] = computer;
		}

		displayBoard(board);
		turn = opponent(turn);
	}

	announceWinner(winner(board), computer, human);
}
