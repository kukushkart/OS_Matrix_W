#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <windows.h>
#include <iomanip>
#include <mutex>
#include <memory>

using namespace std;
using namespace chrono;

const int N = 50;

void initMatrix(vector<vector<double>>& matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = rand() % 10;
        }
    }
}

void clearMatrix(vector<vector<double>>& matrix) {
    for (int i = 0; i < N; i++) {
        fill(matrix[i].begin(), matrix[i].end(), 0.0);
    }
}

void multiplySimple(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void multiplyBlockKernel(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int rowStart, int rowEnd,
    int colStart, int colEnd,
    int innerStart, int innerEnd,
    mutex* mtx,
    CRITICAL_SECTION* cs) {

    int h = rowEnd - rowStart;
    int w = colEnd - colStart;
    vector<vector<double>> localRes(h, vector<double>(w, 0.0));

    for (int i = rowStart; i < rowEnd; i++) {
        for (int j = colStart; j < colEnd; j++) {
            double sum = 0;
            for (int k = innerStart; k < innerEnd; k++) {
                sum += A[i][k] * B[k][j];
            }
            localRes[i - rowStart][j - colStart] = sum;
        }
    }

    if (mtx) mtx->lock();
    if (cs) EnterCriticalSection(cs);

    for (int i = rowStart; i < rowEnd; i++) {
        for (int j = colStart; j < colEnd; j++) {
            C[i][j] += localRes[i - rowStart][j - colStart];
        }
    }

    if (cs) LeaveCriticalSection(cs);
    if (mtx) mtx->unlock();
}

void multiplyThreadStd(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int blockSize) {

    vector<thread> threads;
    int numBlocks = (N + blockSize - 1) / blockSize;

    vector<unique_ptr<mutex>> mutexes;
    for (int i = 0; i < numBlocks * numBlocks; ++i) {
        mutexes.push_back(make_unique<mutex>());
    }

    for (int i = 0; i < numBlocks; i++) {
        for (int j = 0; j < numBlocks; j++) {
            for (int k = 0; k < numBlocks; k++) {

                int rowStart = i * blockSize;
                int rowEnd = min((i + 1) * blockSize, N);

                int colStart = j * blockSize;
                int colEnd = min((j + 1) * blockSize, N);

                int innerStart = k * blockSize;
                int innerEnd = min((k + 1) * blockSize, N);

                mutex* mtx = mutexes[i * numBlocks + j].get();

                threads.emplace_back(multiplyBlockKernel,
                    ref(A), ref(B), ref(C),
                    rowStart, rowEnd, colStart, colEnd, innerStart, innerEnd,
                    mtx, nullptr);
            }
        }
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
}

struct BlockParams {
    const vector<vector<double>>* A;
    const vector<vector<double>>* B;
    vector<vector<double>>* C;
    int rowStart, rowEnd;
    int colStart, colEnd;
    int innerStart, innerEnd;
    CRITICAL_SECTION* cs;
};

DWORD WINAPI multiplyBlockWinWrapper(LPVOID param) {
    BlockParams* p = (BlockParams*)param;
    multiplyBlockKernel(*(p->A), *(p->B), *(p->C),
        p->rowStart, p->rowEnd, p->colStart, p->colEnd,
        p->innerStart, p->innerEnd, nullptr, p->cs);
    delete p;
    return 0;
}

void multiplyThreadWin(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int blockSize) {

    vector<HANDLE> threads;
    int numBlocks = (N + blockSize - 1) / blockSize;

    vector<CRITICAL_SECTION> css(numBlocks * numBlocks);
    for (auto& cs : css) {
        InitializeCriticalSection(&cs);
    }

    for (int i = 0; i < numBlocks; i++) {
        for (int j = 0; j < numBlocks; j++) {
            for (int k = 0; k < numBlocks; k++) {

                BlockParams* p = new BlockParams;
                p->A = &A; p->B = &B; p->C = &C;
                p->rowStart = i * blockSize;
                p->rowEnd = min((i + 1) * blockSize, N);
                p->colStart = j * blockSize;
                p->colEnd = min((j + 1) * blockSize, N);
                p->innerStart = k * blockSize;
                p->innerEnd = min((k + 1) * blockSize, N);
                p->cs = &css[i * numBlocks + j];

                HANDLE hThread = CreateThread(NULL, 0, multiplyBlockWinWrapper, p, 0, NULL);
                if (hThread != NULL) {
                    threads.push_back(hThread);
                }
                else {
                    delete p;
                }
            }
        }
    }

    const int MAX_WAIT = 64;
    for (size_t i = 0; i < threads.size(); i += MAX_WAIT) {
        int count = min((int)(threads.size() - i), MAX_WAIT);
        WaitForMultipleObjects(count, threads.data() + i, TRUE, INFINITE);
    }

    for (auto h : threads) {
        CloseHandle(h);
    }

    for (auto& cs : css) {
        DeleteCriticalSection(&cs);
    }
}

int main() {
    setlocale(LC_ALL, "Russian");
    srand((unsigned int)time(0));

    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> C(N, vector<double>(N));

    initMatrix(A);
    initMatrix(B);

    cout << "Размер матрицы: " << N << "x" << N << endl;
    cout << "Однопоточное умножение " << endl;

    auto start = high_resolution_clock::now();
    multiplySimple(A, B, C);
    auto end = high_resolution_clock::now();
    auto durationSimple = duration_cast<milliseconds>(end - start).count();

    cout << "Время (один поток): " << durationSimple << " мс" << endl << endl;

    cout << left << setw(15) << "Размер блока"
        << setw(15) << "Кол-во потоков"
        << setw(20) << "std::thread (мс)"
        << setw(20) << "WinAPI (мс)"
        << endl;
    cout << endl;

    vector<int> blockSizes;
    for (int k = 1; k <= N; k *= 2) {
        blockSizes.push_back(k);
    }
    if (blockSizes.back() != N) {
        blockSizes.push_back(N);
    }

    for (int blockSize : blockSizes) {
        int numBlocks = (N + blockSize - 1) / blockSize;
        long long totalThreads = (long long)numBlocks * numBlocks * numBlocks;

        if (totalThreads > 5000) {
            cout << left << setw(15) << blockSize
                << setw(15) << totalThreads
                << setw(40) << "Слишком много потоков (skip)" << endl;
            continue;
        }

        clearMatrix(C);
        start = high_resolution_clock::now();
        multiplyThreadStd(A, B, C, blockSize);
        end = high_resolution_clock::now();
        auto timeStd = duration_cast<milliseconds>(end - start).count();

        clearMatrix(C);
        start = high_resolution_clock::now();
        multiplyThreadWin(A, B, C, blockSize);
        end = high_resolution_clock::now();
        auto timeWin = duration_cast<milliseconds>(end - start).count();

        cout << left << setw(15) << blockSize
            << setw(15) << totalThreads
            << setw(20) << timeStd
            << setw(20) << timeWin
            << endl;
    }

    return 0;
}
