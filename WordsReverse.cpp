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

//��ÿ�����ʽ��з�ת
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
	//cout << "������һ���ַ�����";
	char a[] = "I am a student.";
	//���ʵĳ���
	int n = strlen(a);
	cout << "n=" << n << endl;
	//��ʾδ��תǰ���ַ���
	print(a, n);
	//���ַ�����Ϊ�����ļ�������,���ֱ���з�ת
	for (int i = 0; i <= n; i++)
	{
		if (a[i] == ' ' || a[i] == '\0')
		{
			//���ʷ�ת
			reverse(a, i - num, i - 1);
			num = 0;
		}
		else
		{
			num++;
		}
	}
	//�м���
	print(a, n);
	//��ʾ��ת֮����ַ���
	for (int i = n - 1; i >= 0; i--)
	{
		cout << a[i];
	}
	cout << endl;

	return 0;
}
