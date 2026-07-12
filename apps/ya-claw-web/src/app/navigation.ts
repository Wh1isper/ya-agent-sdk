type AppNavigate = (path: string, replace?: boolean) => void

type NavigationRegistration = {
  navigate: AppNavigate
}

const navigationRegistrations: NavigationRegistration[] = []

export function registerAppNavigate(navigate: AppNavigate) {
  const registration = { navigate }
  navigationRegistrations.push(registration)

  return () => {
    const index = navigationRegistrations.indexOf(registration)
    if (index !== -1) navigationRegistrations.splice(index, 1)
  }
}

export function hasRegisteredAppNavigation() {
  return navigationRegistrations.length > 0
}

export function navigateApp(path: string, replace = false) {
  const registration =
    navigationRegistrations[navigationRegistrations.length - 1]
  if (registration) {
    registration.navigate(path, replace)
    return
  }
  if (replace) {
    window.history.replaceState(null, '', path)
  } else {
    window.history.pushState(null, '', path)
  }
}
