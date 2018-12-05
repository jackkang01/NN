#include<iostream>

using namespace std;

void print(char *s, int m)
{
	for (int i = 0; i < m; i++)
	{
		cout << s[i];
	}
	cout << endl;
}

//将每个单词进行翻转
void reverse(char *s, int low, int high)
{
	while (low < high)
	{
		int tmp = s[high];
		s[high] = s[low];
		s[low] = tmp;
		low++;
		high--;
	}
}

int main()
{
	int num = 0;
	int low, high;
	//cout << "请输入一个字符串：";
	char a[] = "I am a student.";
	//单词的长度
	int n = strlen(a);
	cout << "n=" << n << endl;
	//显示未翻转前的字符串
	print(a, n);
	//将字符串分为独立的几个单词,并分别进行翻转
	for (int i = 0; i <= n; i++)
	{
		if (a[i] == ' ' || a[i] == '\0')
		{
			//单词翻转
			reverse(a, i - num, i - 1);
			num = 0;
		}
		else
		{
			num++;
		}
	}
	//中间结果
	print(a, n);
	//显示翻转之后的字符串
	for (int i = n - 1; i >= 0; i--)
	{
		cout << a[i];
	}
	cout << endl;

	return 0;
}
